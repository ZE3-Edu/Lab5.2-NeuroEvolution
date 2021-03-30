import numpy as np
np.seterr(over='ignore', divide='raise')
from matplotlib import pyplot as plt

class SimpleNeuralNet():
    def activation_function(self, x):
        return 1/(1+np.exp(-x))
    
    def deepcopy(self):
        new_net = SimpleNeuralNet(self.num_inputs, self.num_outputs, self.layer_node_counts)
        new_net.layers = [np.copy(layer) for layer in self.layers]
        return new_net
    
    def execute(self, input_vector):
        assert len(input_vector) == self.num_inputs ,\
        "wrong input vector size"

        next_v = input_vector

        # iterate through layers, computing the activation
        # of the weighted inputs from the previous layer
        for layer in self.layers:
            # add a bias to each layer [1]
            next_v = np.append(next_v, 1)
            
            # pump the input vector through the matrix multiplication
            # and our activation function
            next_v = self.activation_function(np.dot(next_v, layer))
            
        return next_v
        
    def __init__(self, num_inputs, num_outputs, layer_node_counts=[]):
        self.num_inputs = num_inputs
        self.layer_node_counts = layer_node_counts
        self.num_outputs = num_outputs
        self.layers = []
        
        last_num_neurons = self.num_inputs
        for nc in layer_node_counts + [num_outputs]:
            # for now, we'll just use random weights in the range [-5,5]
            # +1 handles adding a bias node for each layer of nodes
            self.layers.append(np.random.uniform(-5, 5, size=(last_num_neurons+1, nc)))
            last_num_neurons = nc

def get_network_fitness(simple_net, input_set, target_output_set):
    assert(len(input_set) == len(target_output_set))
    total_distance = 0

    for test_index in range(len(input_set)):
        test_output = simple_net.execute(input_set[test_index])
        target_output = output_set[test_index]

        #typical distance formula between points
        distances = np.linalg.norm(test_output - target_output)

        #sum the distances up and keep a running tab!
        total_distance += np.sum(distances)

    #we actually want fitness to *increase* as things get fitter
    #so we'll just negate this value.
    return -total_distance

def tournament_selection(population, input_set, target_set, fit_func, tournament_size=3):
    sample_pop = np.random.choice(population, size=tournament_size)
    sample_pop_fitness = [fit_func(p, input_set, target_set) for p in sample_pop]
    winner_idx = np.argmax(sample_pop_fitness)
    
    return sample_pop[winner_idx]


# TODO: IMPLEMENT THIS
def mutate_network(simple_net, mutation_rate=0.001):
    for layer_to_mut in simple_net.layers:
        dims = layer_to_mut.shape



"""    
This code will work even without a mutation function, but you'll not be introducing any variation into the population. Once you have a working mutation function, you'll have more fun. 

For this problem, we're going to try to evolve a solution to the last question from the last worksheet. 

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |
"""
input_set = [[0,0],
             [0,1],
             [1,0],
             [1,1]]

output_set = [[0], 
              [1],
              [1], 
              [0]]


pop_size = 100
num_generations = 100
mutation_rate = 1E-3

#Build up a population of SimpleNeuralNets
#with one hidden layer made up of 5 neurons.
population = [ SimpleNeuralNet(num_inputs=2, 
                               num_outputs=1, 
                               layer_node_counts=[5])
              for i in range(pop_size)]


avg_fitnesses = []
for gen in range(num_generations):
    print(gen)

    #do tournament selection to fill up the population
    #with deepcopies of the neural networks. 
    selected_individuals = [tournament_selection(population, 
                                                 input_set, 
                                                 output_set, 
                                                 get_network_fitness).deepcopy()
                            for _ in range(pop_size)]
    #mutate them!
    for individual in selected_individuals:
      mutate_network(individual, mutation_rate)
    
    # calculate mean fitness for each generation
    this_gen_avg_fitness = np.mean([get_network_fitness(idv, input_set, output_set) for idv in selected_individuals])
    
    avg_fitnesses.append(this_gen_avg_fitness)
    
    population = selected_individuals

plt.plot(avg_fitnesses)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()