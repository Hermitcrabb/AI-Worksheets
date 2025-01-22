def calculate_heuristic(start_state, goal_state):
    # Flatten the goal state into a dictionary for reference
    goal_positions = {}
    for stack_index, stack in enumerate(goal_state):
        for block_index, block in enumerate(stack):
            goal_positions[block] = (stack_index, block_index)
    
    heuristic_value = 0
    for stack_index, stack in enumerate(start_state):
        for block_index, block in enumerate(stack):
            # Check if the block is in the goal state
            if block in goal_positions:
                goal_stack, goal_pos = goal_positions[block]
                
                # If the block is at the correct stack and position
                if stack_index == goal_stack and block_index == goal_pos:
                    heuristic_value += 1
                else:
                    heuristic_value -= 1  # Wrong support structure
    
    return heuristic_value

# Input for the given problem
start_state = [['A', 'D', 'C', 'B']]  # Initial configuration
goal_state = [['D', 'C', 'B', 'A']]  # Goal configuration

# Calculate Heuristic Value
heuristic_value = calculate_heuristic(start_state, goal_state)
print("Heuristic Value:",heuristic_value)
