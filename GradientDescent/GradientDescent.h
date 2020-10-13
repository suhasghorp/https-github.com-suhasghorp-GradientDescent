#pragma once

#include <vector>
#include <functional>
#include "autodiff\reverse.hpp"

using namespace autodiff;

class GradientDescent  {
public:
	GradientDescent();
	~GradientDescent();

	void setObjectiveFunction(std::function<double(std::vector<double>)> objFun);

	void setObjectiveFunctionAAD(std::function<var(std::vector<var>)> objFun);

	void setStartPoint(const std::vector<double> startPoint);

	void setStepSize(double stepSize);

	void setMaxIterations(int maxIterations);

	void setUseAAD(bool useAAD);

	void setGradientThreshold(double gradThreshold);

	std::pair< std::vector<double>, double> optimize();

	void printResults(std::pair< std::vector<double>, double> results);

private:
	double computeGradient(int dim);
	std::vector<double> computeGradientVector();
	std::vector<double> computeGradientVectorAAD();
	double computeGradientMagnitude(std::vector<double> gradientVector);

private:
	int m_nDims;
	double m_stepSize;
	int m_maxIter;
	double m_h;
	double m_gradientThreshold;
	std::vector<double> m_startPoint;
	std::vector<double> m_currentPoint;
	std::function<double(std::vector<double>)> m_objFun;
	std::function<var(std::vector<var>)> m_objFunAAD;
	bool m_useAAD;
	

};
