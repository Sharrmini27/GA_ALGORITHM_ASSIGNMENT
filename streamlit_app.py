import streamlit as st
import pandas as pd
import random
import csv
import numpy as np

# ============================================================
#   PART A: GENETIC ALGORITHM ENGINE
# ============================================================

def read_csv_to_dict(file_path):
    """Reads the CSV file and returns a dictionary of ratings."""
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row
            for row in reader:
                if len(row) > 1:
                    program = row[0]
                    try:
                        ratings = [float(x) for x in row[1:]]
                        program_ratings[program] = ratings
                    except ValueError:
                        st.warning(f"Skipping row for '{program}': contains non-numeric rating.")
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Using fallback sample data for testing.")
        program_ratings = {
            'News': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.4],
            'Sports': [0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.4, 0.3],
            'Movies': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.5],
        }
    return program_ratings


def fitness_function(schedule, ratings_data, schedule_length):
    """Calculates the total fitness of a given schedule."""
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data:
            if time_slot < len(ratings_data[program]):
                total_rating += ratings_data[program][time_slot]
    return total_rating


def create_random_schedule(all_programs, schedule_length):
    """Creates a single random schedule."""
    return [random.choice(all_programs) for _ in range(schedule_length)]


def crossover(schedule1, schedule2, schedule_length):
    """Performs single-point crossover."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    crossover_point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2


def mutate(schedule, all_programs, schedule_length):
    """Mutates a schedule by changing one random slot."""
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, schedule_length - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy


def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    """Runs the genetic algorithm optimization."""
    
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule_ever = []
    best_fitness_ever = 0

    for generation in range(generations):
        pop_with_fitness = []
        for schedule in population:
            fitness = fitness_function(schedule, ratings_data, schedule_length)
            pop_with_fitness.append((schedule, fitness))
            
            if fitness > best_fitness_ever:
                best_fitness_ever = fitness
                best_schedule_ever = schedule

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        new_population = []

        # Elitism
        for i in range(elitism_size):
            new_population.append(pop_with_fitness[i][0])

        # Fill rest of population
        while len(new_population) < population_size:
            parent1 = random.choice(pop_with_fitness[:population_size // 2])[0]
            parent2 = random.choice(pop_with_fitness[:population_size // 2])[0]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, schedule_length)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs, schedule_length)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_schedule_ever, best_fitness_ever


# ============================================================
#   PART B: STREAMLIT APPLICATION
# ============================================================

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# Load dataset
file_path = 'program_ratings_updated.csv'
ratings = read_csv_to_dict(file_path)

# Display dataset
st.subheader("📊 Program Ratings Dataset")
try:
    df_display = pd.read_csv(file_path)
    st.dataframe(df_display)
except FileNotFoundError:
    st.error(f"Could not find {file_path} to display.")

if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6:00 - 23:00
    SCHEDULE_LENGTH = len(all_time_slots)

    st.success(f"Loaded {len(all_programs)} programs for optimization.")
    st.info(f"Schedule optimized for {SCHEDULE_LENGTH} hourly slots (6:00 AM - 11:00 PM).")

    # === FIXED PARAMETERS FOR EACH TRIAL ===
    TRIALS = [
        {"name": "Trial 1", "crossover": 0.8, "mutation": 0.2, "seed": 10},
        {"name": "Trial 2", "crossover": 0.9, "mutation": 0.1, "seed": 20},
        {"name": "Trial 3", "crossover": 0.7, "mutation": 0.3, "seed": 30},
    ]

    if st.button("🚀 Run All 3 Trials"):
        for trial in TRIALS:
            random.seed(trial["seed"])
            np.random.seed(trial["seed"])

            st.header(f"{trial['name']} Results")
            st.write(f"**Parameters:** Crossover = {trial['crossover']}, Mutation = {trial['mutation']}")

            best_schedule, best_fitness = genetic_algorithm(
                ratings_data=ratings,
                all_programs=all_programs,
                schedule_length=SCHEDULE_LENGTH,
                crossover_rate=trial["crossover"],
                mutation_rate=trial["mutation"]
            )

            df_result = pd.DataFrame({
                "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
                "Scheduled Program": best_schedule
            })
            st.dataframe(df_result)
            st.write(f"**Best Fitness Score:** {best_fitness:.3f}")
            st.markdown("---")

else:
    st.error("Could not load any program data. Please check the file path and CSV content.")
