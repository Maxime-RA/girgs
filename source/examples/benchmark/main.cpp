
#include <chrono>
#include <iostream>

#include <girgs/Generator.h>


using namespace std;
using namespace girgs;

// GIRG parameter
const auto ple = -2.5;
//const auto alpha = 1.0;
const auto alpha = std::numeric_limits<double>::infinity();
const auto avgDeg = 10;

// measuring parameter
const auto SEED = 13;
const auto RUNS = 50;
const auto N_1 = 1000;
const auto N_2 = 10000;
const auto D_MAX = 4;
using resolution = chrono::milliseconds;


int measure(int dimension, int n, int run) {

    auto weightSeed = SEED;
    auto positionSeed = SEED+run;
    auto samplingSeed = SEED+run+n;

    Generator g;
    g.setWeights(n, ple, weightSeed);
    g.setPositions(n, dimension, positionSeed);
    g.scaleWeights(avgDeg, dimension, alpha);

    // measure
    auto start = chrono::high_resolution_clock::now();
    g.generate(alpha, samplingSeed);
    auto end = chrono::high_resolution_clock::now();

    return chrono::duration_cast<resolution>(end-start).count();
}

int avg(int dimension, int n){
    auto sum = 0;
    for (int i = 0; i < RUNS; ++i)
        sum = sum + measure(dimension, n, i);
    return sum / RUNS;
}


int main(int argc, char* argv[]) {

    for(auto d = 1; d<D_MAX; ++d){
        cout << "d=" << d << '\n';
        for(auto n = N_1; n<N_2; n*=2)
            cout << n << '\t' << avg(d,n) << endl;
        cout << endl;
    }
    return 0;
}
