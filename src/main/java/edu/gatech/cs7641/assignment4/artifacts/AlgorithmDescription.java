package edu.gatech.cs7641.assignment4.artifacts;

public abstract class AlgorithmDescription {

    public Algorithm type;

    public AlgorithmDescription(Algorithm type) {
        this.type = type;
    }

    public abstract String describe();
}
