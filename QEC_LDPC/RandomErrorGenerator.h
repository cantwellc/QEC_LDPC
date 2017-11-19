#pragma once
#include <vector>
#include <random>

class RandomErrorGenerator
{
private:
    std::uniform_int_distribution<int> _indexDistribution;
    std::uniform_int_distribution<int> _errorDistribution;
    std::mt19937 _engine;
public:
    unsigned int seed;
    RandomErrorGenerator(int numVars)
    {
        std::random_device rd; // random seed or mersene twister engine.  could use this exclusively, but mt is faster
        std::mt19937 mt(rd()); // engine to produce random number
        std::uniform_int_distribution<int> indexDist(0, numVars - 1); // distribution for rng of index where errror occurs
        std::uniform_int_distribution<int> errorDist(0, 2); // distribution for rng of error type. x=0, y=1, z=2

        _indexDistribution = indexDist;
        _errorDistribution = errorDist;
        _engine = mt;

        seed = _engine.default_seed;
    }

    ~RandomErrorGenerator()
    {
    }

    void GenerateError(std::vector<int>& xErrors, std::vector<int>& zErrors, int errorWeight)
    {
        // construct random error string
        for (int i = 0; i<errorWeight; ++i)
        {
            // chose the index where an error will occur.
            int index = _indexDistribution(_engine);
            // determine whether the error is x, y, or z.
            int error = _errorDistribution(_engine);
            // set the correct error bits
            if (error == 0 || error == 1) xErrors[index] = 1;
            if (error == 2 || error == 1) zErrors[index] = 1;
        }
    }
};

