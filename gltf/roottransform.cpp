// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <unordered_map>
#include <unordered_set>

namespace
{

// Represents T = scale * rotation: a translation-free similarity transform (uniform scale + rotation).
// Because the scale is uniform, it commutes with rotation, so T = scale * rotation = rotation * scale;
// this is what makes T^-1 = (1/scale) * rotation^T (rotation-inverse) and lets conjugation of a pure
// rotation or a pure (uniform) scale cancel the scale term entirely.
struct RootTransform
{
	float scale;
	float rotation[4]; // unit quaternion, xyzw
};

void quatMultiply(float out[4], const float a[4], const float b[4])
{
	out[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
	out[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
	out[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
	out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}

void quatNormalize(float q[4])
{
	float len = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

	if (len > 0.f)
	{
		float inv = 1.f / len;
		q[0] *= inv, q[1] *= inv, q[2] *= inv, q[3] *= inv;
	}
}

void quatRotateVec(float out[3], const float q[4], const float v[3])
{
	float ux = q[0], uy = q[1], uz = q[2], w = q[3];

	float tx = 2 * (uy * v[2] - uz * v[1]);
	float ty = 2 * (uz * v[0] - ux * v[2]);
	float tz = 2 * (ux * v[1] - uy * v[0]);

	out[0] = v[0] + w * tx + (uy * tz - uz * ty);
	out[1] = v[1] + w * ty + (uz * tx - ux * tz);
	out[2] = v[2] + w * tz + (ux * ty - uy * tx);
}

// p is transformed as T * p = scale * rotate(rotation, p); valid for points and for morph target
// deltas alike since T carries no translation.
void transformPoint(float out[3], const RootTransform& t, const float p[3])
{
	float r[3];
	quatRotateVec(r, t.rotation, p);

	out[0] = r[0] * t.scale;
	out[1] = r[1] * t.scale;
	out[2] = r[2] * t.scale;
}

// directions (normals/tangent axes) transform by the rotation only: T's inverse-transpose is
// (1/scale)*rotation, and since scale > 0 the extra uniform factor doesn't affect direction.
void transformDirection(float out[3], const RootTransform& t, const float d[3])
{
	quatRotateVec(out, t.rotation, d);
}

// q' = rotation * q * rotation^-1 (conjugation): re-expresses q's axis in the new frame while
// preserving its angle; the scale term of T cancels out of a rotation conjugation entirely.
void transformRotation(float out[4], const RootTransform& t, const float q[4])
{
	float conj[4] = {-t.rotation[0], -t.rotation[1], -t.rotation[2], t.rotation[3]};

	float tmp[4];
	quatMultiply(tmp, t.rotation, q);
	quatMultiply(out, tmp, conj);

	quatNormalize(out);
}

void quatToMat3(float m[9], const float q[4])
{
	float x = q[0], y = q[1], z = q[2], w = q[3];
	float x2 = x + x, y2 = y + y, z2 = z + z;
	float xx = x * x2, xy = x * y2, xz = x * z2;
	float yy = y * y2, yz = y * z2, zz = z * z2;
	float wx = w * x2, wy = w * y2, wz = w * z2;

	// column-major 3x3, m[col*3+row]
	m[0] = 1 - (yy + zz), m[1] = xy + wz, m[2] = xz - wy;
	m[3] = xy - wz, m[4] = 1 - (xx + zz), m[5] = yz + wx;
	m[6] = xz + wy, m[7] = yz - wx, m[8] = 1 - (xx + yy);
}

// builds T (inverse=false) or T^-1 (inverse=true) as a full 4x4 column-major matrix, for
// conjugating general affine matrices (inverse bind matrices) that may carry their own
// translation/scale/shear, where the TRS-only helpers above don't apply.
void buildMatrix(float m[16], const RootTransform& t, bool inverse)
{
	float rot[9];
	quatToMat3(rot, t.rotation);

	float scale = inverse ? 1.f / t.scale : t.scale;

	memset(m, 0, 16 * sizeof(float));
	m[15] = 1.f;

	if (inverse)
	{
		// transpose(rotation) * (1/scale)
		m[0] = rot[0] * scale, m[1] = rot[3] * scale, m[2] = rot[6] * scale;
		m[4] = rot[1] * scale, m[5] = rot[4] * scale, m[6] = rot[7] * scale;
		m[8] = rot[2] * scale, m[9] = rot[5] * scale, m[10] = rot[8] * scale;
	}
	else
	{
		m[0] = rot[0] * scale, m[1] = rot[1] * scale, m[2] = rot[2] * scale;
		m[4] = rot[3] * scale, m[5] = rot[4] * scale, m[6] = rot[5] * scale;
		m[8] = rot[6] * scale, m[9] = rot[7] * scale, m[10] = rot[8] * scale;
	}
}

void matrixMultiply(float out[16], const float a[16], const float b[16])
{
	for (int c = 0; c < 4; ++c)
		for (int r = 0; r < 4; ++r)
		{
			float sum = 0.f;

			for (int k = 0; k < 4; ++k)
				sum += a[k * 4 + r] * b[c * 4 + k];

			out[c * 4 + r] = sum;
		}
}

// M' = T * M * T^-1, in place
void conjugateMatrix(float m[16], const RootTransform& t)
{
	float tm[16], ti[16];
	buildMatrix(tm, t, false);
	buildMatrix(ti, t, true);

	float tmp[16], result[16];
	matrixMultiply(tmp, tm, m);
	matrixMultiply(result, tmp, ti);

	memcpy(m, result, sizeof(result));
}

void getLocalTRS(const cgltf_node& node, float t[3], float r[4], float s[3])
{
	if (node.has_matrix)
	{
		decomposeTransform(t, r, s, node.matrix);
	}
	else
	{
		t[0] = node.translation[0], t[1] = node.translation[1], t[2] = node.translation[2];
		r[0] = node.rotation[0], r[1] = node.rotation[1], r[2] = node.rotation[2], r[3] = node.rotation[3];
		s[0] = node.scale[0], s[1] = node.scale[1], s[2] = node.scale[2];
	}
}

void setLocalTRS(cgltf_node& node, const float t[3], const float r[4], const float s[3])
{
	node.has_matrix = false;

	node.has_translation = true;
	memcpy(node.translation, t, 3 * sizeof(float));

	node.has_rotation = true;
	memcpy(node.rotation, r, 4 * sizeof(float));

	node.has_scale = true;
	memcpy(node.scale, s, 3 * sizeof(float));
}

bool isUniformScale(const float s[3], float tol)
{
	float mx = fabsf(s[0]);
	mx = mx < fabsf(s[1]) ? fabsf(s[1]) : mx;
	mx = mx < fabsf(s[2]) ? fabsf(s[2]) : mx;

	if (mx < 1e-8f)
		return true;

	return fabsf(s[0] - s[1]) <= tol * mx && fabsf(s[0] - s[2]) <= tol * mx && fabsf(s[1] - s[2]) <= tol * mx;
}

std::string nodeLabel(const cgltf_data* data, const cgltf_node* node)
{
	if (node->name && *node->name)
		return std::string("'") + node->name + "'";

	char buf[32];
	snprintf(buf, sizeof(buf), "#%d", int(node - data->nodes));

	return std::string(buf);
}

// Recursively conjugates every descendant's local transform by T; records the owning root for
// each visited node so meshes and animation tracks attached anywhere in the subtree can be found
// and transformed the same way afterwards.
void resetDescendant(cgltf_data* data, cgltf_node* node, cgltf_node* root, const RootTransform& t, std::unordered_map<cgltf_node*, cgltf_node*>& owner)
{
	owner[node] = root;

	float lt[3], lr[4], ls[3];
	getLocalTRS(*node, lt, lr, ls);

	if (!isUniformScale(ls, 1e-2f))
		fprintf(stderr, "Warning: node %s has non-uniform scale under a reset scene root transform; results may be incorrect\n", nodeLabel(data, node).c_str());

	float nt[3], nr[4];
	transformPoint(nt, t, lt);
	transformRotation(nr, t, lr);

	setLocalTRS(*node, nt, nr, ls);

	for (cgltf_size i = 0; i < node->children_count; ++i)
		resetDescendant(data, node->children[i], root, t, owner);
}

void bakeMeshStreams(Mesh& mesh, const RootTransform& t)
{
	for (size_t si = 0; si < mesh.streams.size(); ++si)
	{
		Stream& stream = mesh.streams[si];

		if (stream.type == cgltf_attribute_type_position)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
				transformPoint(stream.data[i].f, t, stream.data[i].f);
		}
		else if (stream.type == cgltf_attribute_type_normal)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
				transformDirection(stream.data[i].f, t, stream.data[i].f);
		}
		else if (stream.type == cgltf_attribute_type_tangent)
		{
			for (size_t i = 0; i < stream.data.size(); ++i)
			{
				float xyz[3];
				transformDirection(xyz, t, stream.data[i].f);
				stream.data[i].f[0] = xyz[0], stream.data[i].f[1] = xyz[1], stream.data[i].f[2] = xyz[2];
				// stream.data[i].f[3] (handedness) is unchanged: rotation preserves orientation and scale > 0
			}
		}
	}
}

bool conjugateInverseBindMatrices(const cgltf_skin& skin, const RootTransform& t, std::string& error)
{
	cgltf_accessor* acc = skin.inverse_bind_matrices;

	// no accessor means the spec-mandated default of identity matrices, which conjugation leaves unchanged
	if (!acc)
		return true;

	if (acc->is_sparse || acc->normalized || acc->type != cgltf_type_mat4 || acc->component_type != cgltf_component_type_r_32f || !acc->buffer_view || !acc->buffer_view->buffer)
	{
		error = "skin '" + (skin.name ? std::string(skin.name) : std::string("<unnamed>")) + "' has inverse bind matrices in an unsupported format";
		return false;
	}

	unsigned char* bufferData = static_cast<unsigned char*>(acc->buffer_view->data ? acc->buffer_view->data : acc->buffer_view->buffer->data);
	if (!bufferData)
	{
		error = "skin '" + (skin.name ? std::string(skin.name) : std::string("<unnamed>")) + "' has inverse bind matrices with no backing buffer data";
		return false;
	}

	unsigned char* base = bufferData + acc->buffer_view->offset + acc->offset;
	size_t stride = acc->stride ? acc->stride : 16 * sizeof(float);

	for (size_t i = 0; i < acc->count; ++i)
	{
		float* m = reinterpret_cast<float*>(base + i * stride);
		conjugateMatrix(m, t);
	}

	return true;
}

} // namespace

bool resetSceneRootTransform(cgltf_data* data, std::vector<Mesh>& meshes, std::vector<Animation>& animations, std::string& error)
{
	cgltf_scene* scene = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : NULL);

	if (!scene || scene->nodes_count == 0)
		return true;

	std::unordered_map<cgltf_node*, cgltf_node*> owner; // node -> owning scene root (includes the roots themselves)
	std::unordered_map<cgltf_node*, RootTransform> roots; // root -> T

	const float kTranslationTolerance = 1e-4f;
	const float kScaleTolerance = 1e-3f;

	for (size_t ri = 0; ri < scene->nodes_count; ++ri)
	{
		cgltf_node* root = scene->nodes[ri];

		float t[3], r[4], s[3];
		getLocalTRS(*root, t, r, s);

		if (fabsf(t[0]) > kTranslationTolerance || fabsf(t[1]) > kTranslationTolerance || fabsf(t[2]) > kTranslationTolerance)
		{
			error = "scene root node " + nodeLabel(data, root) + " has a non-zero translation; only translation-free scene roots are supported";
			return false;
		}

		if (!isUniformScale(s, kScaleTolerance) || s[0] <= 1e-6f)
		{
			error = "scene root node " + nodeLabel(data, root) + " does not have a uniform positive scale; only uniform-scale scene roots are supported";
			return false;
		}

		RootTransform rt;
		rt.scale = s[0];
		memcpy(rt.rotation, r, sizeof(rt.rotation));
		quatNormalize(rt.rotation);

		roots[root] = rt;
		owner[root] = root;

		// reset the root's own local transform to identity; its old transform is now T, pushed down into descendants and baked meshes
		float zeroT[3] = {0.f, 0.f, 0.f};
		float identityR[4] = {0.f, 0.f, 0.f, 1.f};
		float oneS[3] = {1.f, 1.f, 1.f};
		setLocalTRS(*root, zeroT, identityR, oneS);

		for (cgltf_size ci = 0; ci < root->children_count; ++ci)
			resetDescendant(data, root->children[ci], root, rt, owner);
	}

	if (roots.empty())
		return true;

	std::unordered_set<cgltf_skin*> processedSkins;

	// splitting a mesh (below) appends newly baked copies to `meshes`; iterate only over the
	// originally parsed entries so a split-off copy is never revisited and baked a second time
	size_t originalMeshCount = meshes.size();

	for (size_t mi = 0; mi < originalMeshCount; ++mi)
	{
		Mesh& mesh = meshes[mi];

		// GPU-instanced meshes are already baked to absolute world-space transforms at parse time
		// (cgltf_node_transform_world was evaluated before this function ever runs), so they are
		// unaffected by resetting node transforms further up the (now bypassed) hierarchy.
		if (!mesh.instances.empty() || mesh.nodes.empty())
			continue;

		// group the mesh's nodes by owning root (or "none" for nodes outside any tracked root)
		std::vector<cgltf_node*> groupOwner;
		std::vector<std::vector<cgltf_node*> > groupNodes;

		for (size_t ni = 0; ni < mesh.nodes.size(); ++ni)
		{
			cgltf_node* node = mesh.nodes[ni];
			std::unordered_map<cgltf_node*, cgltf_node*>::iterator oi = owner.find(node);
			cgltf_node* own = (oi == owner.end()) ? NULL : oi->second;

			size_t gi = 0;
			for (; gi < groupOwner.size(); ++gi)
				if (groupOwner[gi] == own)
					break;

			if (gi == groupOwner.size())
			{
				groupOwner.push_back(own);
				groupNodes.push_back(std::vector<cgltf_node*>());
			}

			groupNodes[gi].push_back(node);
		}

		if (groupOwner.size() == 1 && groupOwner[0] == NULL)
			continue; // mesh entirely outside any tracked root: nothing to do

		Mesh original = (groupOwner.size() > 1) ? mesh : Mesh();

		for (size_t gi = 0; gi < groupOwner.size(); ++gi)
		{
			Mesh* target;

			if (gi == 0)
				target = &meshes[mi];
			else
			{
				meshes.push_back(original);
				target = &meshes.back();
			}

			target->nodes = groupNodes[gi];

			if (groupOwner[gi])
			{
				const RootTransform& t = roots[groupOwner[gi]];

				bakeMeshStreams(*target, t);

				if (target->skin && processedSkins.insert(target->skin).second)
				{
					if (!conjugateInverseBindMatrices(*target->skin, t, error))
						return false;
				}
			}
		}
	}

	for (size_t ai = 0; ai < animations.size(); ++ai)
	{
		Animation& animation = animations[ai];

		for (size_t ti = 0; ti < animation.tracks.size(); ++ti)
		{
			Track& track = animation.tracks[ti];

			std::unordered_map<cgltf_node*, cgltf_node*>::iterator oi = owner.find(track.node);
			if (oi == owner.end())
				continue;

			// the scene root's own animation (if any) isn't part of the "descendants" this function
			// pushes the transform into; its rest pose was reset explicitly above but the (rare) case
			// of an animated scene root is not supported by this transform.
			if (roots.count(track.node))
				continue;

			const RootTransform& t = roots[oi->second];

			switch (track.path)
			{
			case cgltf_animation_path_type_translation:
				for (size_t ki = 0; ki < track.data.size(); ++ki)
					transformPoint(track.data[ki].f, t, track.data[ki].f);
				break;

			case cgltf_animation_path_type_rotation:
				for (size_t ki = 0; ki < track.data.size(); ++ki)
					transformRotation(track.data[ki].f, t, track.data[ki].f);
				break;

			case cgltf_animation_path_type_scale:
			{
				bool warned = false;
				for (size_t ki = 0; ki < track.data.size() && !warned; ++ki)
					if (!isUniformScale(track.data[ki].f, 1e-2f))
					{
						fprintf(stderr, "Warning: node %s has a non-uniform scale animation under a reset scene root transform; results may be incorrect\n", nodeLabel(data, track.node).c_str());
						warned = true;
					}
				// scale keyframes are left unchanged: conjugation of a uniform scale is the identity operation
				break;
			}

			default:
				// weights and other paths are untouched
				break;
			}
		}
	}

	return true;
}
