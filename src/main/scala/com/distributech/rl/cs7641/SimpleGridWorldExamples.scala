package com.distributech.rl.cs7641

import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration
import burlap.mdp.core.action.Action
import burlap.statehashing.discretized.DiscretizingHashableStateFactory

import scala.collection.immutable
import burlap.behavior.policy.PolicyUtils
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration

import scala.collection.JavaConversions._


object SimpleGridWorldExamples extends App {

  import burlap.domain.singleagent.mountaincar.MountainCar

  val mcGen = new MountainCar
  val domain = mcGen.generateDomain
  val s = mcGen.valleyState

  println("Value iteration")
  val hashingFactory = new DiscretizingHashableStateFactory(0.01)
  hashingFactory.addFloorDiscretizingMultipleFor(MountainCar.ATT_X, 0.01)
  hashingFactory.addFloorDiscretizingMultipleFor(MountainCar.ATT_V, 0.0001)
  println((1 to 6).map(
    iter => {
      val iteration = new ValueIteration(domain, 0.99, hashingFactory, 0.0001, iter * 10)
      val policy: GreedyQPolicy = iteration.planFromState(s)
      import burlap.behavior.policy.PolicyUtils
      import burlap.behavior.singleagent.Episode
      val episode = PolicyUtils.rollout(policy, s, domain.getModel, 10000)
      episode.rewardSequence.toList.map(_.toDouble).sum
    }
  ))

  println("Policy iteration")
  println((1 to 6).map(
    iter => {
      val iteration = new PolicyIteration(domain, 0.99, hashingFactory, 0.0001, 1000, iter * 10)
      val policy: GreedyQPolicy = iteration.planFromState(s)
      import burlap.behavior.policy.PolicyUtils
      import burlap.behavior.singleagent.Episode
      val episode = PolicyUtils.rollout(policy, s, domain.getModel, 10000)
      episode.rewardSequence.toList.map(_.toDouble).sum
    }
  ))

  println("Policy iteration")
  println((1 to 6).map(
    iter => {
      val iteration = new PolicyIteration(domain, 0.99, hashingFactory, 0.0001, 1000, iter * 10)
      val policy: GreedyQPolicy = iteration.planFromState(s)
      import burlap.behavior.policy.PolicyUtils
      import burlap.behavior.singleagent.Episode
      val episode = PolicyUtils.rollout(policy, s, domain.getModel, 10000)
      episode.rewardSequence.toList.map(_.toDouble).sum
    }
  ))


}
