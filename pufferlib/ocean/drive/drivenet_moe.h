#ifndef DRIVENET_MOE_H
#define DRIVENET_MOE_H

// Note: drive.h and puffernet.h must be included before this header
// (typically via drivenet.h which is included first in visualize.c)

#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NN_INPUT_SIZE 64
#define NN_HIDDEN_SIZE 256
#define NUM_EXPERTS 3
#define LORA_RANK 8
#define LORA_ALPHA 4.0f
#define ROUTER_HIDDEN_DIM 64

typedef struct DriveNetMoE DriveNetMoE;
struct DriveNetMoE {
    int num_agents;
    int ego_dim;
    float *obs_self;
    float *obs_partner;
    float *obs_road;
    float *partner_linear_output;
    float *road_linear_output;
    float *partner_layernorm_output;
    float *road_layernorm_output;
    float *partner_linear_output_two;
    float *road_linear_output_two;

    // Encoders (same structure as baseline)
    Linear *ego_encoder;
    Linear *road_encoder;
    Linear *partner_encoder;
    LayerNorm *ego_layernorm;
    LayerNorm *road_layernorm;
    LayerNorm *partner_layernorm;
    Linear *ego_encoder_two;
    Linear *road_encoder_two;
    Linear *partner_encoder_two;

    MaxDim1 *partner_max;
    MaxDim1 *road_max;
    CatDim1 *cat1;
    CatDim1 *cat2;
    GELU *gelu;
    Linear *shared_embedding;
    ReLU *relu;
    LSTM *lstm;

    // Router network
    Linear *router_linear1;
    LayerNorm *router_layernorm;
    ReLU *router_relu;
    Linear *router_linear2;
    float *router_hidden;
    float *router_logits;
    float *expert_probs;

    // Actor with LoRA
    Linear *actor_base;
    float *expert_A;  // (NUM_EXPERTS, LORA_RANK, NN_HIDDEN_SIZE)
    float *expert_B;  // (NUM_EXPERTS, action_size, LORA_RANK)
    float *mixed_A;   // (num_agents, LORA_RANK, NN_HIDDEN_SIZE)
    float *mixed_B;   // (num_agents, action_size, LORA_RANK)
    float *lora_intermediate;  // (num_agents, LORA_RANK)
    float *lora_output;  // (num_agents, action_size)
    float *actor_output;  // (num_agents, action_size)
    int action_size;

    Linear *value_fn;
    Multidiscrete *multidiscrete;
};

// Softmax for expert probabilities
void softmax_expert(float *logits, float *probs, int num_agents, int num_experts) {
    for (int b = 0; b < num_agents; b++) {
        float max_val = logits[b * num_experts];
        for (int e = 1; e < num_experts; e++) {
            if (logits[b * num_experts + e] > max_val) {
                max_val = logits[b * num_experts + e];
            }
        }

        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            probs[b * num_experts + e] = expf(logits[b * num_experts + e] - max_val);
            sum += probs[b * num_experts + e];
        }

        for (int e = 0; e < num_experts; e++) {
            probs[b * num_experts + e] /= sum;
        }
    }
}

DriveNetMoE *init_drivenet_moe(Weights *weights, int num_agents, int dynamics_model) {
    DriveNetMoE *net = (DriveNetMoE *)calloc(1, sizeof(DriveNetMoE));

    int ego_dim = (dynamics_model == JERK) ? EGO_FEATURES_JERK : EGO_FEATURES_CLASSIC;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int input_size = NN_INPUT_SIZE;
    int hidden_size = NN_HIDDEN_SIZE;
    int road_feat_onehot = road_features + 6;

    int action_size, logit_sizes[2];
    int action_dim;
    if (dynamics_model == CLASSIC) {
        action_size = 7 * 13;
        logit_sizes[0] = 7 * 13;
        action_dim = 1;
    } else {
        action_size = 4 * 3;
        logit_sizes[0] = 4 * 3;
        action_dim = 1;
    }

    net->num_agents = num_agents;
    net->ego_dim = ego_dim;
    net->action_size = action_size;

    // Observation buffers
    net->obs_self = (float *)calloc(num_agents * ego_dim, sizeof(float));
    net->obs_partner = (float *)calloc(num_agents * max_partners * partner_features, sizeof(float));
    net->obs_road = (float *)calloc(num_agents * max_road_obs * road_feat_onehot, sizeof(float));
    net->partner_linear_output = (float *)calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output = (float *)calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_linear_output_two = (float *)calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_linear_output_two = (float *)calloc(num_agents * max_road_obs * input_size, sizeof(float));
    net->partner_layernorm_output = (float *)calloc(num_agents * max_partners * input_size, sizeof(float));
    net->road_layernorm_output = (float *)calloc(num_agents * max_road_obs * input_size, sizeof(float));

    // Encoders - loaded in DriveMoE parameter order:
    // ego_encoder: Linear(ego_dim, 64), LayerNorm(64), Linear(64, 64)
    net->ego_encoder = make_linear(weights, num_agents, ego_dim, input_size);
    net->ego_layernorm = make_layernorm(weights, num_agents, input_size);
    net->ego_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // road_encoder: Linear(13, 64), LayerNorm(64), Linear(64, 64)
    net->road_encoder = make_linear(weights, num_agents, road_feat_onehot, input_size);
    net->road_layernorm = make_layernorm(weights, num_agents, input_size);
    net->road_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // partner_encoder: Linear(7, 64), LayerNorm(64), Linear(64, 64)
    net->partner_encoder = make_linear(weights, num_agents, partner_features, input_size);
    net->partner_layernorm = make_layernorm(weights, num_agents, input_size);
    net->partner_encoder_two = make_linear(weights, num_agents, input_size, input_size);

    // Max pooling and concatenation
    net->partner_max = make_max_dim1(num_agents, max_partners, input_size);
    net->road_max = make_max_dim1(num_agents, max_road_obs, input_size);
    net->cat1 = make_cat_dim1(num_agents, input_size, input_size);
    net->cat2 = make_cat_dim1(num_agents, input_size + input_size, input_size);
    net->gelu = make_gelu(num_agents, 3 * input_size);

    // shared_embedding: GELU (no params), Linear(192, 256)
    net->shared_embedding = make_linear(weights, num_agents, input_size * 3, hidden_size);
    net->relu = make_relu(num_agents, hidden_size);

    // Router: Linear(192, 64), LayerNorm(64), ReLU, Dropout (no params), Linear(64, 3)
    net->router_linear1 = make_linear(weights, num_agents, input_size * 3, ROUTER_HIDDEN_DIM);
    net->router_layernorm = make_layernorm(weights, num_agents, ROUTER_HIDDEN_DIM);
    net->router_relu = make_relu(num_agents, ROUTER_HIDDEN_DIM);
    net->router_linear2 = make_linear(weights, num_agents, ROUTER_HIDDEN_DIM, NUM_EXPERTS);
    net->router_hidden = (float *)calloc(num_agents * ROUTER_HIDDEN_DIM, sizeof(float));
    net->router_logits = (float *)calloc(num_agents * NUM_EXPERTS, sizeof(float));
    net->expert_probs = (float *)calloc(num_agents * NUM_EXPERTS, sizeof(float));

    // Actor with LoRA: base weight (91, 256), bias (91), expert_A (3, 8, 256), expert_B (3, 91, 8)
    net->actor_base = make_linear(weights, num_agents, hidden_size, action_size);

    // Load LoRA expert weights
    int expert_A_size = NUM_EXPERTS * LORA_RANK * hidden_size;
    int expert_B_size = NUM_EXPERTS * action_size * LORA_RANK;
    net->expert_A = get_weights(weights, expert_A_size);
    net->expert_B = get_weights(weights, expert_B_size);

    // Allocate intermediate buffers for LoRA computation
    net->mixed_A = (float *)calloc(num_agents * LORA_RANK * hidden_size, sizeof(float));
    net->mixed_B = (float *)calloc(num_agents * action_size * LORA_RANK, sizeof(float));
    net->lora_intermediate = (float *)calloc(num_agents * LORA_RANK, sizeof(float));
    net->lora_output = (float *)calloc(num_agents * action_size, sizeof(float));
    net->actor_output = (float *)calloc(num_agents * action_size, sizeof(float));

    // Value function
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);

    // LSTM
    net->lstm = make_lstm(weights, num_agents, hidden_size, NN_HIDDEN_SIZE);
    memset(net->lstm->state_h, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));
    memset(net->lstm->state_c, 0, num_agents * NN_HIDDEN_SIZE * sizeof(float));

    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, action_dim);

    return net;
}

void free_drivenet_moe(DriveNetMoE *net) {
    free(net->obs_self);
    free(net->obs_partner);
    free(net->obs_road);
    free(net->partner_linear_output);
    free(net->road_linear_output);
    free(net->partner_linear_output_two);
    free(net->road_linear_output_two);
    free(net->partner_layernorm_output);
    free(net->road_layernorm_output);
    free(net->ego_encoder);
    free(net->road_encoder);
    free(net->partner_encoder);
    free(net->ego_layernorm);
    free(net->road_layernorm);
    free(net->partner_layernorm);
    free(net->ego_encoder_two);
    free(net->road_encoder_two);
    free(net->partner_encoder_two);
    free(net->partner_max);
    free(net->road_max);
    free(net->cat1);
    free(net->cat2);
    free(net->gelu);
    free(net->shared_embedding);
    free(net->relu);
    free(net->router_linear1);
    free(net->router_layernorm);
    free(net->router_relu);
    free(net->router_linear2);
    free(net->router_hidden);
    free(net->router_logits);
    free(net->expert_probs);
    free(net->actor_base);
    free(net->mixed_A);
    free(net->mixed_B);
    free(net->lora_intermediate);
    free(net->lora_output);
    free(net->actor_output);
    free(net->multidiscrete);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward_moe(DriveNetMoE *net, float *observations, int *actions) {
    int ego_dim = net->ego_dim;
    int max_partners = MAX_AGENTS - 1;
    int max_road_obs = MAX_ROAD_SEGMENT_OBSERVATIONS;
    int partner_features = PARTNER_FEATURES;
    int road_features = ROAD_FEATURES;
    int road_feat_onehot = road_features + 6;
    int hidden_size = NN_HIDDEN_SIZE;
    int action_size = net->action_size;
    float lora_scaling = LORA_ALPHA / LORA_RANK;

    // Clear previous observations
    memset(net->obs_self, 0, net->num_agents * ego_dim * sizeof(float));
    memset(net->obs_partner, 0, net->num_agents * max_partners * partner_features * sizeof(float));
    memset(net->obs_road, 0, net->num_agents * max_road_obs * road_feat_onehot * sizeof(float));

    // Parse observations (same as baseline)
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (ego_dim + max_partners * partner_features + max_road_obs * road_features);
        int partner_offset = b_offset + ego_dim;
        int road_offset = b_offset + ego_dim + max_partners * partner_features;

        for (int i = 0; i < ego_dim; i++) {
            net->obs_self[b * ego_dim + i] = observations[b_offset + i];
        }

        for (int i = 0; i < max_partners; i++) {
            for (int j = 0; j < partner_features; j++) {
                net->obs_partner[b * max_partners * partner_features + i * partner_features + j] =
                    observations[partner_offset + i * partner_features + j];
            }
        }

        for (int i = 0; i < MAX_ROAD_SEGMENT_OBSERVATIONS; i++) {
            for (int j = 0; j < 7; j++) {
                net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT + j] =
                    observations[road_offset + i * 7 + j];
            }
            for (int j = 0; j < 7; j++) {
                if (j == (int)observations[road_offset + i * 7 + 6]) {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT + 6 + j] = 1.0f;
                } else {
                    net->obs_road[b * MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES_ONEHOT + i * ROAD_FEATURES_ONEHOT + 6 + j] = 0.0f;
                }
            }
        }
    }

    // Ego encoder forward pass
    linear(net->ego_encoder, net->obs_self);
    layernorm(net->ego_layernorm, net->ego_encoder->output);
    linear(net->ego_encoder_two, net->ego_layernorm->output);

    // Partner encoder forward pass (per-object processing)
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *obj_features = &net->obs_partner[b * max_partners * partner_features + obj * partner_features];
            _linear(obj_features, net->partner_encoder->weights, net->partner_encoder->bias,
                    &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    partner_features, NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *after_first = &net->partner_linear_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->partner_layernorm->weights, net->partner_layernorm->bias,
                       &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_partners; obj++) {
            float *obj_features = &net->partner_layernorm_output[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(obj_features, net->partner_encoder_two->weights, net->partner_encoder_two->bias,
                    &net->partner_linear_output_two[b * max_partners * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // Road encoder forward pass (per-object processing)
    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *obj_features = &net->obs_road[b * max_road_obs * ROAD_FEATURES_ONEHOT + obj * ROAD_FEATURES_ONEHOT];
            _linear(obj_features, net->road_encoder->weights, net->road_encoder->bias,
                    &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    ROAD_FEATURES_ONEHOT, NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_linear_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _layernorm(after_first, net->road_layernorm->weights, net->road_layernorm->bias,
                       &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                       NN_INPUT_SIZE);
        }
    }

    for (int b = 0; b < net->num_agents; b++) {
        for (int obj = 0; obj < max_road_obs; obj++) {
            float *after_first = &net->road_layernorm_output[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE];
            _linear(after_first, net->road_encoder_two->weights, net->road_encoder_two->bias,
                    &net->road_linear_output_two[b * max_road_obs * NN_INPUT_SIZE + obj * NN_INPUT_SIZE], 1,
                    NN_INPUT_SIZE, NN_INPUT_SIZE);
        }
    }

    // Max pooling and concatenation
    max_dim1(net->partner_max, net->partner_linear_output_two);
    max_dim1(net->road_max, net->road_linear_output_two);
    cat_dim1(net->cat1, net->ego_encoder_two->output, net->road_max->output);
    cat_dim1(net->cat2, net->cat1->output, net->partner_max->output);

    // Store concat_features for router (before GELU)
    float *concat_features = net->cat2->output;

    // Shared embedding: GELU -> Linear
    gelu(net->gelu, concat_features);
    linear(net->shared_embedding, net->gelu->output);
    relu(net->relu, net->shared_embedding->output);

    // LSTM
    lstm(net->lstm, net->relu->output);

    // Router forward pass on concat_features (pre-GELU)
    linear(net->router_linear1, concat_features);
    layernorm(net->router_layernorm, net->router_linear1->output);
    relu(net->router_relu, net->router_layernorm->output);
    linear(net->router_linear2, net->router_relu->output);
    softmax_expert(net->router_linear2->output, net->expert_probs, net->num_agents, NUM_EXPERTS);

    // Actor with LoRA: base_out + (x @ mixed_A @ mixed_B) * scaling
    // First compute base output
    linear(net->actor_base, net->lstm->state_h);

    // Mix expert weights based on router output
    // mixed_A[b,r,i] = sum_e expert_probs[b,e] * expert_A[e,r,i]
    // mixed_B[b,o,r] = sum_e expert_probs[b,e] * expert_B[e,o,r]
    memset(net->mixed_A, 0, net->num_agents * LORA_RANK * hidden_size * sizeof(float));
    memset(net->mixed_B, 0, net->num_agents * action_size * LORA_RANK * sizeof(float));

    for (int b = 0; b < net->num_agents; b++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float prob = net->expert_probs[b * NUM_EXPERTS + e];
            // Mix A: (rank, hidden_size)
            for (int r = 0; r < LORA_RANK; r++) {
                for (int i = 0; i < hidden_size; i++) {
                    net->mixed_A[b * LORA_RANK * hidden_size + r * hidden_size + i] +=
                        prob * net->expert_A[e * LORA_RANK * hidden_size + r * hidden_size + i];
                }
            }
            // Mix B: (action_size, rank)
            for (int o = 0; o < action_size; o++) {
                for (int r = 0; r < LORA_RANK; r++) {
                    net->mixed_B[b * action_size * LORA_RANK + o * LORA_RANK + r] +=
                        prob * net->expert_B[e * action_size * LORA_RANK + o * LORA_RANK + r];
                }
            }
        }
    }

    // Compute LoRA output: x @ A.T @ B.T
    // lora_intermediate[b,r] = sum_i x[b,i] * mixed_A[b,r,i]
    memset(net->lora_intermediate, 0, net->num_agents * LORA_RANK * sizeof(float));
    for (int b = 0; b < net->num_agents; b++) {
        for (int r = 0; r < LORA_RANK; r++) {
            for (int i = 0; i < hidden_size; i++) {
                net->lora_intermediate[b * LORA_RANK + r] +=
                    net->lstm->state_h[b * hidden_size + i] *
                    net->mixed_A[b * LORA_RANK * hidden_size + r * hidden_size + i];
            }
        }
    }

    // lora_output[b,o] = sum_r lora_intermediate[b,r] * mixed_B[b,o,r]
    memset(net->lora_output, 0, net->num_agents * action_size * sizeof(float));
    for (int b = 0; b < net->num_agents; b++) {
        for (int o = 0; o < action_size; o++) {
            for (int r = 0; r < LORA_RANK; r++) {
                net->lora_output[b * action_size + o] +=
                    net->lora_intermediate[b * LORA_RANK + r] *
                    net->mixed_B[b * action_size * LORA_RANK + o * LORA_RANK + r];
            }
        }
    }

    // Combine: actor_output = base_out + lora_output * scaling
    for (int b = 0; b < net->num_agents; b++) {
        for (int o = 0; o < action_size; o++) {
            net->actor_output[b * action_size + o] =
                net->actor_base->output[b * action_size + o] +
                net->lora_output[b * action_size + o] * lora_scaling;
        }
    }

    // Value function
    linear(net->value_fn, net->lstm->state_h);

    // Get action by taking argmax of actor output
    softmax_multidiscrete(net->multidiscrete, net->actor_output, actions);
}

#endif // DRIVENET_MOE_H
