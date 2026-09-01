// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// This file exercises resetSceneRootTransform (gltf/roottransform.cpp) against a hand-built
// cgltf hierarchy covering a static mesh branch, an animated branch (independent translation/
// rotation/scale channels) and a skinned mesh branch. Rather than re-deriving the transform's
// internal conjugation formulas, each check compares world-space results computed via cgltf's
// own cgltf_node_transform_world/local before and after the reset: since the whole point of the
// transform is that it moves the scene root's transform into the hierarchy/vertices without
// changing anything visually, "before" and "after" world results must match exactly (up to
// floating point error).

namespace
{

void check(bool condition, const char* expr, int line)
{
	if (!condition)
	{
		fprintf(stderr, "testResetSceneRootTransform: check failed (%s) at gltf/roottransformtest.cpp:%d\n", expr, line);
		assert(!"resetSceneRootTransform test check failed");
	}
}

#define TT_CHECK(x) check((x), #x, __LINE__)

float lerp(float a, float b, float t)
{
	return a + (b - a) * t;
}

void normalize4(float q[4])
{
	float len = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	if (len > 0.f)
	{
		q[0] /= len, q[1] /= len, q[2] /= len, q[3] /= len;
	}
}

void lerpQuat(float out[4], const float a[4], const float b[4], float t)
{
	out[0] = lerp(a[0], b[0], t);
	out[1] = lerp(a[1], b[1], t);
	out[2] = lerp(a[2], b[2], t);
	out[3] = lerp(a[3], b[3], t);
	normalize4(out);
}

void axisAngleQuat(float q[4], float ax, float ay, float az, float angle)
{
	float len = sqrtf(ax * ax + ay * ay + az * az);
	ax /= len, ay /= len, az /= len;

	float h = angle * 0.5f;
	float s = sinf(h);

	q[0] = ax * s, q[1] = ay * s, q[2] = az * s, q[3] = cosf(h);
}

// column-major 4x4 multiply, out = a*b
void matMul4(float out[16], const float a[16], const float b[16])
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

void transformPointM(float out[3], const float m[16], const float p[3])
{
	out[0] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
	out[1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
	out[2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
}

// direction/delta: linear part only, no translation
void transformDirM(float out[3], const float m[16], const float d[3])
{
	out[0] = m[0] * d[0] + m[4] * d[1] + m[8] * d[2];
	out[1] = m[1] * d[0] + m[5] * d[1] + m[9] * d[2];
	out[2] = m[2] * d[0] + m[6] * d[1] + m[10] * d[2];
}

void normalize3(float v[3])
{
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 0.f)
	{
		v[0] /= len, v[1] /= len, v[2] /= len;
	}
}

bool nearEqualV(const float* a, const float* b, int n, float eps)
{
	for (int i = 0; i < n; ++i)
		if (fabsf(a[i] - b[i]) > eps)
			return false;
	return true;
}

const Stream* findStream(const Mesh& mesh, cgltf_attribute_type type, int target)
{
	for (size_t i = 0; i < mesh.streams.size(); ++i)
		if (mesh.streams[i].type == type && mesh.streams[i].target == target)
			return &mesh.streams[i];
	return NULL;
}

Attr makeAttr(float x, float y, float z, float w = 0.f)
{
	Attr a;
	a.f[0] = x, a.f[1] = y, a.f[2] = z, a.f[3] = w;
	return a;
}

const float kEps = 5e-3f;

} // namespace

void testResetSceneRootTransform()
{
	// ---- hierarchy ----
	// 0 = R (scene root, translation-free uniform-scale transform to be reset)
	// 1 = A (static mesh branch, child of R)
	// 2 = B (animated branch, child of R; no mesh, translation/rotation/scale each animated independently)
	// 3 = C (skinned mesh branch, child of R)
	// 4 = J1 (skeleton root joint, child of R)
	// 5 = J2 (child joint, child of J1)
	cgltf_node nodes[6];
	memset(nodes, 0, sizeof(nodes));

	// R
	nodes[0].has_translation = true;
	nodes[0].has_rotation = true;
	axisAngleQuat(nodes[0].rotation, 1.f, 2.f, 3.f, 0.9f);
	nodes[0].has_scale = true;
	nodes[0].scale[0] = nodes[0].scale[1] = nodes[0].scale[2] = 1.7f;

	// A
	nodes[1].parent = &nodes[0];
	nodes[1].has_translation = true;
	nodes[1].translation[0] = 1.f, nodes[1].translation[1] = 2.f, nodes[1].translation[2] = 3.f;
	nodes[1].has_rotation = true;
	axisAngleQuat(nodes[1].rotation, 0.f, 0.f, 1.f, 0.7f);
	nodes[1].has_scale = true;
	nodes[1].scale[0] = nodes[1].scale[1] = nodes[1].scale[2] = 1.f;

	// B
	nodes[2].parent = &nodes[0];
	nodes[2].has_translation = true;
	nodes[2].translation[0] = 5.f;
	nodes[2].has_rotation = true;
	nodes[2].rotation[3] = 1.f;
	nodes[2].has_scale = true;
	nodes[2].scale[0] = nodes[2].scale[1] = nodes[2].scale[2] = 1.f;

	// C
	nodes[3].parent = &nodes[0];
	nodes[3].has_translation = true;
	nodes[3].translation[0] = -2.f, nodes[3].translation[1] = 1.f;
	nodes[3].has_rotation = true;
	nodes[3].rotation[3] = 1.f;
	nodes[3].has_scale = true;
	nodes[3].scale[0] = nodes[3].scale[1] = nodes[3].scale[2] = 1.f;

	// J1
	nodes[4].parent = &nodes[0];
	nodes[4].has_translation = true;
	nodes[4].translation[1] = 3.f;
	nodes[4].has_rotation = true;
	axisAngleQuat(nodes[4].rotation, 1.f, 0.f, 0.f, 0.4f);
	nodes[4].has_scale = true;
	nodes[4].scale[0] = nodes[4].scale[1] = nodes[4].scale[2] = 1.f;

	// J2
	nodes[5].parent = &nodes[4];
	nodes[5].has_translation = true;
	nodes[5].translation[1] = 1.f;
	nodes[5].has_rotation = true;
	nodes[5].rotation[3] = 1.f;
	nodes[5].has_scale = true;
	nodes[5].scale[0] = nodes[5].scale[1] = nodes[5].scale[2] = 1.f;

	cgltf_node* rChildren[4] = {&nodes[1], &nodes[2], &nodes[3], &nodes[4]};
	nodes[0].children = rChildren;
	nodes[0].children_count = 4;

	cgltf_node* j1Children[1] = {&nodes[5]};
	nodes[4].children = j1Children;
	nodes[4].children_count = 1;

	cgltf_node* sceneNodes[1] = {&nodes[0]};
	cgltf_scene scene;
	memset(&scene, 0, sizeof(scene));
	scene.nodes = sceneNodes;
	scene.nodes_count = 1;

	cgltf_data data;
	memset(&data, 0, sizeof(data));
	data.nodes = nodes;
	data.nodes_count = 6;
	data.scenes = &scene;
	data.scenes_count = 1;
	data.scene = &scene;

	// ---- static branch mesh (A) ----
	float posAData[3][3] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
	float nrmAData[3][3] = {{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}};
	float morphAData[3][3] = {{0.1f, 0.2f, -0.3f}, {0.5f, 0.f, 0.f}, {0.f, 0.4f, 0.1f}};

	Stream posA;
	posA.type = cgltf_attribute_type_position, posA.index = 0, posA.target = 0, posA.custom_name = NULL;
	Stream nrmA;
	nrmA.type = cgltf_attribute_type_normal, nrmA.index = 0, nrmA.target = 0, nrmA.custom_name = NULL;
	Stream morphA;
	morphA.type = cgltf_attribute_type_position, morphA.index = 0, morphA.target = 1, morphA.custom_name = NULL;

	for (int i = 0; i < 3; ++i)
	{
		posA.data.push_back(makeAttr(posAData[i][0], posAData[i][1], posAData[i][2]));
		nrmA.data.push_back(makeAttr(nrmAData[i][0], nrmAData[i][1], nrmAData[i][2]));
		morphA.data.push_back(makeAttr(morphAData[i][0], morphAData[i][1], morphAData[i][2]));
	}

	Mesh meshA;
	meshA.scene = 0;
	meshA.nodes.push_back(&nodes[1]);
	meshA.material = NULL;
	meshA.skin = NULL;
	memset(&meshA.extras, 0, sizeof(meshA.extras));
	meshA.type = cgltf_primitive_type_triangles;
	meshA.streams.push_back(posA);
	meshA.streams.push_back(nrmA);
	meshA.streams.push_back(morphA);
	meshA.geometry_duplicate = false;
	meshA.targets = 1;
	meshA.quality = 1.f;

	// ---- skinned branch mesh (C) ----
	float posCData[2][3] = {{0.2f, 0.f, 0.f}, {-0.1f, 0.3f, 0.2f}};

	Stream posC;
	posC.type = cgltf_attribute_type_position, posC.index = 0, posC.target = 0, posC.custom_name = NULL;
	for (int i = 0; i < 2; ++i)
		posC.data.push_back(makeAttr(posCData[i][0], posCData[i][1], posCData[i][2]));

	cgltf_node ibmNode0;
	memset(&ibmNode0, 0, sizeof(ibmNode0));
	ibmNode0.has_translation = true;
	ibmNode0.translation[0] = 0.3f, ibmNode0.translation[1] = -0.1f, ibmNode0.translation[2] = 0.2f;
	ibmNode0.has_rotation = true;
	axisAngleQuat(ibmNode0.rotation, 0.f, 1.f, 0.f, 0.5f);
	ibmNode0.has_scale = true;
	ibmNode0.scale[0] = ibmNode0.scale[1] = ibmNode0.scale[2] = 1.f;

	cgltf_node ibmNode1 = ibmNode0;
	ibmNode1.translation[0] = -0.2f, ibmNode1.translation[2] = 0.4f;

	float ibmData[2][16];
	cgltf_node_transform_local(&ibmNode0, ibmData[0]);
	cgltf_node_transform_local(&ibmNode1, ibmData[1]);

	cgltf_buffer ibmBuffer;
	memset(&ibmBuffer, 0, sizeof(ibmBuffer));
	ibmBuffer.data = ibmData;
	ibmBuffer.size = sizeof(ibmData);

	cgltf_buffer_view ibmView;
	memset(&ibmView, 0, sizeof(ibmView));
	ibmView.buffer = &ibmBuffer;
	ibmView.offset = 0;
	ibmView.size = sizeof(ibmData);

	cgltf_accessor ibmAccessor;
	memset(&ibmAccessor, 0, sizeof(ibmAccessor));
	ibmAccessor.component_type = cgltf_component_type_r_32f;
	ibmAccessor.type = cgltf_type_mat4;
	ibmAccessor.count = 2;
	ibmAccessor.stride = 16 * sizeof(float);
	ibmAccessor.buffer_view = &ibmView;

	cgltf_node* joints[2] = {&nodes[4], &nodes[5]};
	cgltf_skin skin;
	memset(&skin, 0, sizeof(skin));
	skin.joints = joints;
	skin.joints_count = 2;
	skin.inverse_bind_matrices = &ibmAccessor;

	Mesh meshC;
	meshC.scene = 0;
	meshC.nodes.push_back(&nodes[3]);
	meshC.material = NULL;
	meshC.skin = &skin;
	memset(&meshC.extras, 0, sizeof(meshC.extras));
	meshC.type = cgltf_primitive_type_triangles;
	meshC.streams.push_back(posC);
	meshC.geometry_duplicate = false;
	meshC.targets = 0;
	meshC.quality = 1.f;

	// ---- animated branch mesh (B) ----
	// a node's own conjugated world transform alone is not visually meaningful in isolation (it
	// picks up a trailing T^-1 that only cancels against a baked mesh vertex or an inverse bind
	// matrix downstream), so - as with the static branch - verification needs a mesh attached to
	// the animated node to check the composed (world transform * baked vertex) invariant.
	float posBData[3] = {1.f, 0.f, 0.f};

	Stream posB;
	posB.type = cgltf_attribute_type_position, posB.index = 0, posB.target = 0, posB.custom_name = NULL;
	posB.data.push_back(makeAttr(posBData[0], posBData[1], posBData[2]));

	Mesh meshB;
	meshB.scene = 0;
	meshB.nodes.push_back(&nodes[2]);
	meshB.material = NULL;
	meshB.skin = NULL;
	memset(&meshB.extras, 0, sizeof(meshB.extras));
	meshB.type = cgltf_primitive_type_triangles;
	meshB.streams.push_back(posB);
	meshB.geometry_duplicate = false;
	meshB.targets = 0;
	meshB.quality = 1.f;

	std::vector<Mesh> meshes;
	meshes.push_back(meshA);
	meshes.push_back(meshC);
	meshes.push_back(meshB);
	size_t meshAIndex = 0, meshCIndex = 1, meshBIndex = 2;

	// ---- animated branch (B): independent translation/rotation/scale channels ----
	Attr trans0 = makeAttr(0.f, 0.f, 0.f);
	Attr trans1 = makeAttr(10.f, -4.f, 2.f);

	Attr rot0 = makeAttr(0.f, 0.f, 0.f, 1.f);
	float rot1q[4];
	axisAngleQuat(rot1q, 0.f, 0.f, 1.f, 1.2f);
	Attr rot1 = makeAttr(rot1q[0], rot1q[1], rot1q[2], rot1q[3]);

	Attr scale0 = makeAttr(1.f, 1.f, 1.f);
	Attr scale1 = makeAttr(2.f, 2.f, 2.f);

	Track trackT = Track();
	trackT.node = &nodes[2];
	trackT.path = cgltf_animation_path_type_translation;
	trackT.components = 1;
	trackT.interpolation = cgltf_interpolation_type_linear;
	trackT.time.push_back(0.f), trackT.time.push_back(1.f);
	trackT.data.push_back(trans0), trackT.data.push_back(trans1);

	Track trackR = Track();
	trackR.node = &nodes[2];
	trackR.path = cgltf_animation_path_type_rotation;
	trackR.components = 1;
	trackR.interpolation = cgltf_interpolation_type_linear;
	trackR.time.push_back(0.f), trackR.time.push_back(1.f);
	trackR.data.push_back(rot0), trackR.data.push_back(rot1);

	Track trackS = Track();
	trackS.node = &nodes[2];
	trackS.path = cgltf_animation_path_type_scale;
	trackS.components = 1;
	trackS.interpolation = cgltf_interpolation_type_linear;
	trackS.time.push_back(0.f), trackS.time.push_back(1.f);
	trackS.data.push_back(scale0), trackS.data.push_back(scale1);

	Animation anim;
	anim.name = "test";
	anim.start = 0.f;
	anim.frames = 0;
	anim.tracks.push_back(trackT);
	anim.tracks.push_back(trackR);
	anim.tracks.push_back(trackS);

	std::vector<Animation> animations;
	animations.push_back(anim);

	const float sampleTimes[] = {0.f, 0.25f, 0.5f, 0.75f, 1.f};
	const int sampleCount = 5;

	// ---- capture "before" ground truth from the original (unmodified) hierarchy ----

	float worldABefore[16];
	cgltf_node_transform_world(&nodes[1], worldABefore);

	float posBBefore[sampleCount][3];
	for (int i = 0; i < sampleCount; ++i)
	{
		float t = sampleTimes[i];
		float trn[3] = {lerp(trans0.f[0], trans1.f[0], t), lerp(trans0.f[1], trans1.f[1], t), lerp(trans0.f[2], trans1.f[2], t)};
		float rot[4];
		lerpQuat(rot, rot0.f, rot1.f, t);
		float scl[3] = {lerp(scale0.f[0], scale1.f[0], t), lerp(scale0.f[1], scale1.f[1], t), lerp(scale0.f[2], scale1.f[2], t)};

		memcpy(nodes[2].translation, trn, sizeof(trn));
		memcpy(nodes[2].rotation, rot, sizeof(rot));
		memcpy(nodes[2].scale, scl, sizeof(scl));

		float worldBBefore[16];
		cgltf_node_transform_world(&nodes[2], worldBBefore);
		transformPointM(posBBefore[i], worldBBefore, posBData);
	}
	// restore B's rest pose
	nodes[2].translation[0] = 5.f, nodes[2].translation[1] = 0.f, nodes[2].translation[2] = 0.f;
	nodes[2].rotation[0] = 0.f, nodes[2].rotation[1] = 0.f, nodes[2].rotation[2] = 0.f, nodes[2].rotation[3] = 1.f;
	nodes[2].scale[0] = nodes[2].scale[1] = nodes[2].scale[2] = 1.f;

	float jointWorldBefore[2][16];
	cgltf_node_transform_world(&nodes[4], jointWorldBefore[0]);
	cgltf_node_transform_world(&nodes[5], jointWorldBefore[1]);

	float skinMatrixBefore[2][16];
	matMul4(skinMatrixBefore[0], jointWorldBefore[0], ibmData[0]);
	matMul4(skinMatrixBefore[1], jointWorldBefore[1], ibmData[1]);

	float skinPosBefore[2][3];
	transformPointM(skinPosBefore[0], skinMatrixBefore[0], posCData[0]);
	transformPointM(skinPosBefore[1], skinMatrixBefore[1], posCData[0]);

	// ---- run the transform under test ----

	std::string error;
	bool ok = resetSceneRootTransform(&data, meshes, animations, error);
	TT_CHECK(ok);

	// ---- static branch: baked mesh vertices composed with the (now-identity-rooted) world chain must match the original ----

	float worldAAfter[16];
	cgltf_node_transform_world(&nodes[1], worldAAfter);

	const Stream* posAAfter = findStream(meshes[meshAIndex], cgltf_attribute_type_position, 0);
	const Stream* nrmAAfter = findStream(meshes[meshAIndex], cgltf_attribute_type_normal, 0);
	const Stream* morphAAfter = findStream(meshes[meshAIndex], cgltf_attribute_type_position, 1);
	TT_CHECK(posAAfter != NULL && nrmAAfter != NULL && morphAAfter != NULL);

	for (int i = 0; i < 3; ++i)
	{
		float expected[3], actual[3];

		transformPointM(expected, worldABefore, posAData[i]);
		transformPointM(actual, worldAAfter, posAAfter->data[i].f);
		TT_CHECK(nearEqualV(expected, actual, 3, kEps));

		float expectedN[3], actualN[3];
		transformDirM(expectedN, worldABefore, nrmAData[i]);
		normalize3(expectedN);
		transformDirM(actualN, worldAAfter, nrmAAfter->data[i].f);
		normalize3(actualN);
		TT_CHECK(nearEqualV(expectedN, actualN, 3, kEps));

		float expectedM[3], actualM[3];
		transformDirM(expectedM, worldABefore, morphAData[i]);
		transformDirM(actualM, worldAAfter, morphAAfter->data[i].f);
		TT_CHECK(nearEqualV(expectedM, actualM, 3, kEps));
	}

	// ---- animated branch: sampling the (transformed) tracks at the same times, composed with the
	// baked mesh vertex, must reproduce the original (pre-reset) world positions ----

	// anim.tracks/animations were built from copies of trackT/trackR/trackS above, so the data
	// resetSceneRootTransform actually mutated in place lives in animations[0].tracks[0..2]
	const Track& animTrackT = animations[0].tracks[0];
	const Track& animTrackR = animations[0].tracks[1];
	const Track& animTrackS = animations[0].tracks[2];

	const Stream* posBAfter = findStream(meshes[meshBIndex], cgltf_attribute_type_position, 0);
	TT_CHECK(posBAfter != NULL);

	for (int i = 0; i < sampleCount; ++i)
	{
		float t = sampleTimes[i];
		float trn[3] = {lerp(animTrackT.data[0].f[0], animTrackT.data[1].f[0], t), lerp(animTrackT.data[0].f[1], animTrackT.data[1].f[1], t), lerp(animTrackT.data[0].f[2], animTrackT.data[1].f[2], t)};
		float rot[4];
		lerpQuat(rot, animTrackR.data[0].f, animTrackR.data[1].f, t);
		float scl[3] = {lerp(animTrackS.data[0].f[0], animTrackS.data[1].f[0], t), lerp(animTrackS.data[0].f[1], animTrackS.data[1].f[1], t), lerp(animTrackS.data[0].f[2], animTrackS.data[1].f[2], t)};

		memcpy(nodes[2].translation, trn, sizeof(trn));
		memcpy(nodes[2].rotation, rot, sizeof(rot));
		memcpy(nodes[2].scale, scl, sizeof(scl));

		float worldAfter[16];
		cgltf_node_transform_world(&nodes[2], worldAfter);

		float posAfter[3];
		transformPointM(posAfter, worldAfter, posBAfter->data[0].f);

		TT_CHECK(nearEqualV(posBBefore[i], posAfter, 3, kEps));
	}

	// ---- skinned branch: joint world * inverse bind, applied to the baked vertex, must reproduce the original skinned position ----

	float jointWorldAfter[2][16];
	cgltf_node_transform_world(&nodes[4], jointWorldAfter[0]);
	cgltf_node_transform_world(&nodes[5], jointWorldAfter[1]);

	float ibmAfter[2][16];
	TT_CHECK(cgltf_accessor_read_float(&ibmAccessor, 0, ibmAfter[0], 16) != 0);
	TT_CHECK(cgltf_accessor_read_float(&ibmAccessor, 1, ibmAfter[1], 16) != 0);

	const Stream* posCAfter = findStream(meshes[meshCIndex], cgltf_attribute_type_position, 0);
	TT_CHECK(posCAfter != NULL);

	for (int j = 0; j < 2; ++j)
	{
		float skinMatrixAfter[16];
		matMul4(skinMatrixAfter, jointWorldAfter[j], ibmAfter[j]);

		float posAfter[3];
		transformPointM(posAfter, skinMatrixAfter, posCAfter->data[0].f);

		TT_CHECK(nearEqualV(skinPosBefore[j], posAfter, 3, kEps));
	}
}
