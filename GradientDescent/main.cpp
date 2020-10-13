#include "GradientDescent.h"
#include <iostream>
#include <functional>
#include "autodiff\reverse.hpp"

using namespace autodiff;

int main(int argc, char* argv[]) {
	GradientDescent solver;
	
	
	auto objFunc1 = [](std::vector<double> x) {return x[0] * x[0]; };
	solver.setObjectiveFunction(objFunc1);
	std::vector<double> startPoint = { -1.0 };
	solver.setUseAAD(false);
	solver.setStartPoint(startPoint);
	solver.setMaxIterations(50);
	solver.setStepSize(0.2); 	
	std::pair< std::vector<double>, double> results = solver.optimize();
	std::cout << "f(x) = x ^2 " << std::endl;
	std::cout << "Without AAD : Approximate Derivatives " << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;

	auto objFuncAAD1 = [](std::vector<var> x) {return x[0] * x[0]; };
	solver.setObjectiveFunctionAAD(objFuncAAD1);
	startPoint = { -1.0 };
	solver.setUseAAD(true);
	solver.setStartPoint(startPoint);
	solver.setMaxIterations(50);
	solver.setStepSize(0.2);
	results = solver.optimize();
	std::cout << "f(x) = x^2" << std::endl;
	std::cout << "With AAD : Exact Derivatives " << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;
	

	auto objFunc2 = [](std::vector<double> x) {return (x[0] * x[0] * x[0] * x[0]) + (2 * (x[0] * x[0] * x[0])) - (6 * (x[0] * x[0])) + (4 * x[0]) + 2; };
	solver.setObjectiveFunction(objFunc2);
	startPoint = { -4.0 };
	solver.setUseAAD(false);
	solver.setStartPoint(startPoint);
	solver.setStepSize(0.04);
	results = solver.optimize();
	std::cout << "f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2" << std::endl;
	std::cout << "Without AAD : Approximate Derivatives" << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;

	auto objFunc2AAD = [](std::vector<autodiff::var> x) {return (x[0] * x[0] * x[0] * x[0]) + (2 * (x[0] * x[0] * x[0])) - (6 * (x[0] * x[0])) + (4 * x[0]) + 2; };
	solver.setObjectiveFunctionAAD(objFunc2AAD);
	startPoint = { -4.0 };
	solver.setUseAAD(true);
	solver.setStartPoint(startPoint);
	solver.setStepSize(0.04);
	results = solver.optimize();
	std::cout << "f(x) = x^4 + 2x^3 - 6x^2 + 4x + 2" << std::endl;
	std::cout << "With AAD : Exact Derivatives " << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;

	auto objFunc3 = [](std::vector<double> x) {return (x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1]); };
	solver.setObjectiveFunction(objFunc3);
	startPoint = { 5.0,5.0 };
	solver.setUseAAD(false);
	solver.setStartPoint(startPoint);
	solver.setMaxIterations(50);
	solver.setStepSize(0.1);
	results = solver.optimize();
	std::cout << "f(x,y) = x^2 + xy + y^2" << std::endl;
	std::cout << "Without AAD : Approximate Derivatives" << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;

	auto objFunc3AAD = [](std::vector<var> x) {return (x[0] * x[0]) + (x[0] * x[1]) + (x[1] * x[1]); };
	solver.setObjectiveFunctionAAD(objFunc3AAD);
	startPoint = { 5.0,5.0 };
	solver.setUseAAD(true);
	solver.setStartPoint(startPoint);
	solver.setMaxIterations(50);
	solver.setStepSize(0.1);
	results = solver.optimize();
	std::cout << "f(x,y) = x^2 + xy + y^2" << std::endl;
	std::cout << "With AAD : Exact Derivatives" << std::endl;
	solver.printResults(results);
	std::cout << "---------------------------------------" << std::endl;


	return 0;
}