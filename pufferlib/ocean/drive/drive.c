#include "drivenet.h"
#include "drivenet_moe.h"
#include <string.h>

// Use this test if the network changes to ensure that the forward pass
// matches the torch implementation to the 3rd or ideally 4th decimal place
void test_drivenet() {
    int num_obs = 1848;
    int num_actions = 2;
    int num_agents = 4;

    float *observations = calloc(num_agents * num_obs, sizeof(float));
    for (int i = 0; i < num_obs * num_agents; i++) {
        observations[i] = i % 7;
    }

    int *actions = calloc(num_agents * num_actions, sizeof(int));

    // Weights* weights = load_weights("resources/drive/puffer_drive_weights.bin");
    Weights *weights = load_weights("puffer_drive_weights.bin");
    DriveNet *net = init_drivenet(weights, num_agents, CLASSIC);

    forward(net, observations, actions);
    for (int i = 0; i < num_agents * num_actions; i++) {
        printf("idx: %d, action: %d, logits:", i, actions[i]);
        for (int j = 0; j < num_actions; j++) {
            printf(" %.6f", net->actor->output[i * num_actions + j]);
        }
        printf("\n");
    }
    free_drivenet(net);
    free(weights);
}

void demo(const char *map_name, const char *policy_name, int use_moe) {
    // Note: The settings below match the MoE training config (puffer_drive_moe.ini)
    Drive env = {
        .human_agent_idx = 0,
        .action_type = 0,           // Discrete
        .dynamics_model = CLASSIC,  // Classic dynamics
        .reward_vehicle_collision = -0.5f,
        .reward_offroad_collision = -0.5f,
        .reward_goal = 1.0f,
        .reward_goal_post_respawn = 0.25f,
        .goal_radius = 2.0f,
        .goal_behavior = 0,         // Respawn on goal (matching training)
        .goal_target_distance = 30.0f,
        .goal_speed = 100.0f,       // Matching training config
        .dt = 0.1f,
        .episode_length = 91,       // Matching training config
        .termination_mode = 1,      // Matching training config
        .collision_behavior = 0,
        .offroad_behavior = 0,
        .init_steps = 0,
        .init_mode = INIT_ALL_VALID,
        .control_mode = CONTROL_VEHICLES,
        .map_name = map_name,
    };
    allocate(&env);

    // Check if we have any active agents
    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: No active agents found in map '%s'\n", map_name);
        fprintf(stderr, "  Total objects: %d, Actors created: %d\n", env.num_objects, env.num_actors);
        free_allocated(&env);
        return;
    }

    // Validate human_agent_idx
    if (env.human_agent_idx >= env.active_agent_count) {
        printf("Warning: human_agent_idx (%d) >= active_agent_count (%d), setting to 0\n",
               env.human_agent_idx, env.active_agent_count);
        env.human_agent_idx = 0;
    }

    printf("Map loaded: %d active agents, %d static agents, %d total actors\n",
           env.active_agent_count, env.static_agent_count, env.num_actors);

    c_reset(&env);
    c_render(&env);

    // Load weights and initialize network
    Weights *weights = load_weights(policy_name);
    DriveNet *net = NULL;
    DriveNetMoE *net_moe = NULL;

    if (use_moe) {
        printf("Using MoE network (DriveNetMoE)\n");
        net_moe = init_drivenet_moe(weights, env.active_agent_count, env.dynamics_model);
    } else {
        printf("Using baseline network (DriveNet)\n");
        net = init_drivenet(weights, env.active_agent_count, env.dynamics_model);
    }

    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        int *actions = (int *)env.actions; // Single integer per agent

        // Forward pass through appropriate network
        if (use_moe) {
            forward_moe(net_moe, env.observations, actions);
        } else {
            forward(net, env.observations, actions);
        }

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (env.dynamics_model == CLASSIC) {
                // Classic dynamics: acceleration and steering
                int accel_idx = 3; // neutral (0 m/s²)
                int steer_idx = 6; // neutral (0.0 steering)

                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    accel_idx += accel_delta;
                    if (accel_idx > 6)
                        accel_idx = 6;
                }
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    accel_idx -= accel_delta;
                    if (accel_idx < 0)
                        accel_idx = 0;
                }
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    steer_idx += steer_delta; // Increase steering index for left turn
                    if (steer_idx > 12)
                        steer_idx = 12;
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    steer_idx -= steer_delta; // Decrease steering index for right turn
                    if (steer_idx < 0)
                        steer_idx = 0;
                }

                // Encode into single integer: action = accel_idx * 13 + steer_idx
                actions[env.human_agent_idx] = accel_idx * 13 + steer_idx;

            } else if (env.dynamics_model == JERK) {
                // Jerk dynamics: longitudinal and lateral jerk
                // JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f}
                // JERK_LAT[3] = {-4.0f, 0.0f, 4.0f}
                int jerk_long_idx = 2; // neutral (0.0)
                int jerk_lat_idx = 1;  // neutral (0.0)

                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                    jerk_long_idx = 3; // acceleration (4.0)
                }
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                    jerk_long_idx = 0; // hard braking (-15.0)
                }
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    jerk_lat_idx = 2; // left turn (4.0)
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    jerk_lat_idx = 0; // right turn (-4.0)
                }

                // Encode into single integer: action = jerk_long_idx * 3 + jerk_lat_idx
                actions[env.human_agent_idx] = jerk_long_idx * 3 + jerk_lat_idx;
            }
        }

        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    if (use_moe) {
        free_drivenet_moe(net_moe);
    } else {
        free_drivenet(net);
    }
    free(weights);
}

void performance_test() {

    long test_time = 10;
    Drive env = {
        .human_agent_idx = 0,
        .dynamics_model = CLASSIC, // Classic dynamics
        .action_type = 0,          // Discrete
        .map_name = "resources/drive/binaries/map_000.bin",
        .dt = 0.1f,
        .init_steps = 0,
    };
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    allocate(&env);
    c_reset(&env);
    end_time = clock();
    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Init time: %f\n", cpu_time_used);

    long start = time(NULL);
    int i = 0;
    int (*actions)[2] = (int (*)[2])env.actions;

    while (time(NULL) - start < test_time) {
        // Set random actions for all agents
        for (int j = 0; j < env.active_agent_count; j++) {
            int accel = rand() % 7;
            int steer = rand() % 13;
            actions[j][0] = accel; // -1, 0, or 1
            actions[j][1] = steer; // Random steering
        }

        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i * env.active_agent_count) / (end - start));
    free_allocated(&env);
}

void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --map-name <path>     Path to map file (default: resources/drive/binaries/training/map_000.bin)\n");
    printf("  --policy-name <path>  Path to weights file (default: resources/drive/puffer_drive_weights.bin)\n");
    printf("  --moe                 Use MoE network (DriveNetMoE) instead of baseline (DriveNet)\n");
    printf("  --num-maps <n>        Number of maps for random selection (default: 100)\n");
    printf("  --help                Show this help message\n");
    printf("\nControls:\n");
    printf("  Hold SHIFT + Arrow keys/WASD to manually control the human agent\n");
}

int main(int argc, char *argv[]) {
    // Default values
    const char *map_name = NULL;
    const char *policy_name = "resources/drive/puffer_drive_weights.bin";
    const char *map_dir = "resources/drive/binaries/training";
    int use_moe = 0;
    int num_maps = 100;
    char map_buffer[512];

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--map-name") == 0) {
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --map-name requires a path argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name requires a path argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--moe") == 0) {
            use_moe = 1;
        } else if (strcmp(argv[i], "--num-maps") == 0) {
            if (i + 1 < argc) {
                num_maps = atoi(argv[i + 1]);
                i++;
            } else {
                fprintf(stderr, "Error: --num-maps requires a number argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--map-dir") == 0) {
            if (i + 1 < argc) {
                map_dir = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --map-dir requires a path argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // If no map specified, pick a random one from map_dir
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "%s/map_%03d.bin", map_dir, random_map);
        map_name = map_buffer;
        printf("Using random map: %s\n", map_name);
    } else {
        printf("Using specified map: %s\n", map_name);
    }

    printf("Using policy: %s\n", policy_name);
    printf("MoE mode: %s\n", use_moe ? "enabled" : "disabled");

    demo(map_name, policy_name, use_moe);

    return 0;
}
