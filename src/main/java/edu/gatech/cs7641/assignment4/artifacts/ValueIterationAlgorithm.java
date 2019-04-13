package edu.gatech.cs7641.assignment4.artifacts;

public class ValueIterationAlgorithm extends AlgorithmDescription {
    public double gamma;
    public double maxDelta;

    public ValueIterationAlgorithm(double gamma, double maxDelta) {
        super(Algorithm.ValueIteration);
        this.gamma = gamma;
        this.maxDelta = maxDelta;
    }

    @Override
    public String describe() {
        return "ValueIteration_gamma_" + gamma + "_maxDelta_" + maxDelta;
    }
}
