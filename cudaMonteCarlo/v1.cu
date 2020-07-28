/*
// Monte carlo approach to solving the dice game craps.
//
// Rules
// -----
// Roll 2 dice and sum values.
//
// Part 1: First Roll
// * Win if roll 7 or 11
// * Loose if roll 2, 3, or 12
// * Go onto Part 2 if did not win or loose.
//
// Part 2: Roll until win/loose
// * Win if re-roll number from Part 1
// * Loose if roll 7
*/
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"


__global__ void roll2(int* out) {
	/*
	Roll two dice.

	curand_uniform, range = (0, 1]

	Stable over ????? threads.
	*/
	int threadx = threadIdx.x + (threadIdx.y * 512);

	curandState state;
	curand_init((unsigned long long)clock() + threadx, 0, 0, &state);

	int die1 = (int) ceil(curand_uniform(&state) * 6);
	int die2 = (int) ceil(curand_uniform(&state) * 6);

	out[threadx] = die1 + die2;
}


__global__ void craps_part1(int* rolls_curr, bool* game_state, bool* outcomes) {
	/*
	First roll of game of craps, update wins and losses.
	*/
	int threadx = threadIdx.x + (threadIdx.y * 512);

	if (!game_state[threadx])
		return;

	int value = rolls_curr[threadx];

	if (value == 7 || value == 11) {
		game_state[threadx] = false;
		outcomes[threadx] = true;
	} else if (value == 2 || value == 3 || value == 12) {
		game_state[threadx] = false;
	}
}


__global__ void craps_part2(int* rolls_curr, int* rolls_original, bool* game_state, bool* outcomes) {
	int threadx = threadIdx.x + (threadIdx.y * 512);

	if (!game_state[threadx])
		return;

	int value = rolls_curr[threadx];

	if (value == rolls_original[threadx]) {
		game_state[threadx] = false;
		outcomes[threadx] = true;
	}
	else if (value == 7) {
		game_state[threadx] = false;
	}
}


int main(void) {
	unsigned int BLOCKS = 1;
	unsigned int GAME_PER_BLOCK = 512;
	unsigned int N_GAMES = BLOCKS * GAME_PER_BLOCK;
	bool game_active = true;

	unsigned int wins_total = 0;
	bool* c_outcomes = new bool[N_GAMES], * c_states = new bool[N_GAMES];

	bool* all_false = new bool[N_GAMES], *all_true = new bool[N_GAMES];
	for (int i = 0; i < N_GAMES; i++) {
		all_false[i] = false;
		all_true[i] = true;
	}

	bool* game_states = nullptr;
	cudaMalloc((void**)&game_states, (int)N_GAMES * sizeof(bool));
	cudaMemcpy(game_states, all_true, (int)N_GAMES * sizeof(bool), cudaMemcpyHostToDevice);
	bool* outcomes = nullptr;
	cudaMalloc((void**)&outcomes, (int)N_GAMES * sizeof(bool));
	cudaMemcpy(outcomes, all_false, (int)N_GAMES * sizeof(bool), cudaMemcpyHostToDevice);

	//
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	//// Part 1
	int* rolls_part1 = nullptr;
	cudaMalloc((void**)&rolls_part1, (int)N_GAMES * sizeof(int));

	roll2 << <BLOCKS, GAME_PER_BLOCK >> > (rolls_part1);
	craps_part1 << <BLOCKS, GAME_PER_BLOCK >> > (rolls_part1, game_states, outcomes);
	cudaDeviceSynchronize();

	cudaMemcpy(c_outcomes, outcomes, (int)N_GAMES * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N_GAMES; i++)
		wins_total += (int)c_outcomes[i];

	//// Part 2
	int* rolls_part2 = nullptr;
	cudaMalloc((void**)&rolls_part2, (int)N_GAMES * sizeof(int));

	game_active = false;
	cudaMemcpy(c_states, game_states, (int)N_GAMES * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N_GAMES; i++) {
		if(c_states[i]) {
			game_active = true;
			break;
		}
	}

	while (game_active) {
		cudaMemcpy(outcomes, all_false, (int)N_GAMES * sizeof(bool), cudaMemcpyHostToDevice);

		roll2 << <BLOCKS, GAME_PER_BLOCK >> > (rolls_part2);
		craps_part2 << <BLOCKS, GAME_PER_BLOCK >> > (rolls_part2, rolls_part1, game_states, outcomes);
		cudaDeviceSynchronize();  // Adds latency

		cudaMemcpy(c_outcomes, outcomes, (int)N_GAMES * sizeof(bool), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N_GAMES; i++)
			wins_total += (int)c_outcomes[i];
		
		game_active = false;
		cudaMemcpy(c_states, game_states, (int)N_GAMES * sizeof(bool), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N_GAMES; i++) {
			if (c_states[i]) {
				game_active = true;
				break;
			}
		}
	}

	//
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	printf(
		"Win pct: %0.3f Time: %5.dms\n",
		wins_total / (float)N_GAMES,
		std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
	);

	return EXIT_SUCCESS;
}
