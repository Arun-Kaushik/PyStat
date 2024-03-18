import numpy as np

def simulate_markov_chain(T, n_steps, start_state=None):
  """
  Simulates a Markov chain for a given number of steps.

  Args:
      T: Transition Probability Matrix (numpy array).
      n_steps: Number of steps to simulate.
      start_state (optional): Index of the starting state. If not provided,
          a random state will be chosen.

  Returns:
      A list of states visited during the simulation.
  """

  # Check if TPM is valid (square matrix and sums to 1 in each row)
  if not np.allclose(T.sum(axis=1), 1):
    raise ValueError("Invalid Transition Probability Matrix. Rows must sum to 1.")

  # Get the number of states
  n_states = T.shape[0]

  # Choose the starting state
  if start_state is None:
    current_state = np.random.choice(n_states)
  else:
    current_state = start_state

  # Initialize list to store states
  states = [current_state]

  # Simulate transitions
  for _ in range(n_steps):
    # Get transition probabilities for the current state
    transition_probs = T[current_state]

    # Sample the next state based on transition probabilities
    next_state = np.random.choice(n_states, p=transition_probs)

    # Update current state and store in list
    current_state = next_state
    states.append(current_state)

  return states

# Example usage
T = np.array([
    [0.4, 0.3, 0.3],
    [0.2, 0.5, 0.3],
    [0.1, 0.4, 0.5]
])

# Simulate 10 steps starting from state 1
states = simulate_markov_chain(T, 10, start_state=1)

# Print the list of visited states
print(states)
