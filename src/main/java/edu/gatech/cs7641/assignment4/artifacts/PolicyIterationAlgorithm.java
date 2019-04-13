package edu.gatech.cs7641.assignment4.artifacts;

public class PolicyIterationAlgorithm extends AlgorithmDescription {
    public double gamma;
    public double maxDelta;
    public int numberOfInteralIterations;

    public PolicyIterationAlgorithm(double gamma, double maxDelta, int numberOfInteralIterations) {
        super(Algorithm.PolicyIteration);
        this.gamma = gamma;
        this.maxDelta = maxDelta;

        this.numberOfInteralIterations = numberOfInteralIterations;
    }

    @Override
    public String describe() {
        return "PolicyIteration_gamma_" + gamma + "_maxDelta_" + maxDelta + "_numberOfInteralIterations_" + numberOfInteralIterations;
    }

}
