package edu.gatech.cs7641.assignment4.artifacts;

import burlap.behavior.policy.Policy;

public class QLearningAlgorithm extends AlgorithmDescription {
    public double gamma;
    public double learningRate;
    public Policy learningPolicy;

    public QLearningAlgorithm(double gamma, double learningRate, Policy learningPolicy) {
        super(Algorithm.QLearning);
        this.gamma = gamma;
        this.learningRate = learningRate;
        this.learningPolicy = learningPolicy;
    }

    @Override
    public String describe() {
        return "PolicyIteration_gamma_" + gamma + "_learningRate_" + learningRate + "_learningPolicy_" + learningPolicy.getClass().getSimpleName();
    }

}
