import numpy as np
from simulator import Simulator
from agents import Wasp, Larvae

def gaussian_attraction(x,y,x0,y0,spread,peak):
    """
    Evaluates a 2D Gaussian function at positions (x,y) with
    mean (x0,y0) and spread. The function returns the value
    of the Gaussian at the given positions, scaled by the
    factor 'peak'.
    """
    return peak*np.exp(-((x-x0)**2 + (y-y0)**2)/spread)

def estimate_gradient(points, z, edge_order=2):
    """
    points: (N,2) array with columns (x,y) from a full rectangular grid (flattened).
    z:      (N,) array of scalar values aligned with 'points' rows.
    Returns:
        dzdx_flat, dzdy_flat of shape (N,), aligned with the input row order.
    """
    x = points[:,0].astype(float)
    y = points[:,1].astype(float)
    z = np.asarray(z, dtype=float).ravel()
    N = x.size

    # recover grid axes and indices
    xs, inv_x = np.unique(x, return_inverse=True)
    ys, inv_y = np.unique(y, return_inverse=True)
    nx, ny = xs.size, ys.size
    if nx * ny != N:
        raise ValueError("Input does not cover a full rectangular grid.")

    # check duplicates/missing points
    counting = np.zeros((ny, nx), dtype=int)
    counting[inv_y, inv_x] += 1
    if not np.all(counting == 1):
        raise ValueError("Grid has duplicates or missing points.")

    # place z onto grid (rows=y, cols=x), then differentiate
    Z = np.empty((ny, nx), float)
    Z[inv_y, inv_x] = z
    dZdy, dZdx = np.gradient(Z, ys, xs, edge_order=edge_order)  # note: returns (∂/∂y, ∂/∂x)

    # map back to original flat ordering
    return dZdx[inv_y, inv_x], dZdy[inv_y, inv_x]


class instanceGenerator():
    def __init__(self,larvae_to_wasps_ratio:float=0.3,total_number_of_larvae:int=0,percentage_foragers:float =0.1,min_number_of_cells:int = 30, max_number_of_cells:int = 100, \
                 nest_fill_percentage:float = 0.5, forage_fill_percentage:float = 0.1, larvae_hunger_multiplier:float = 2.0, mean_food_capacity:float = 10.0, std_food_capacity:float = 2.0, forage_distance:int = 5):
        
        self.larvae_to_wasps_ratio = larvae_to_wasps_ratio
        self.total_number_of_larvae = total_number_of_larvae
        self.percentage_foragers = percentage_foragers
        self.min_number_of_cells = min_number_of_cells
        self.max_number_of_cells = max_number_of_cells
        self.nest_fill_percentage = nest_fill_percentage
        self.mean_food_capacity = mean_food_capacity
        self.std_food_capacity = std_food_capacity
        self.forage_distance = forage_distance
        self.forage_fill_percentage = forage_fill_percentage
        self.larvae_hunger_multiplier = larvae_hunger_multiplier

    def generateWasps(self):
        wasps = []
        return wasps
    def generateLarvae(self,x,y,hunger_multiplier=1.0):
        random_food = np.random.choice([0.9,3,5],1)
        random_hunger = 1 if random_food == 0.9 else 2
        agent_id = "L"+str(x)+str(y) 
        random_hunger_multiplier = np.random.choice([1.0,hunger_multiplier],1)
        return Larvae(agent_id, x=x, y=y, hunger=random_hunger, food=random_food,hungerMultiplier=random_hunger_multiplier)
    def generateGrid(self,radius):
        grid_x, grid_y = np.mgrid[-radius:radius+1, -radius:radius+1]
        grid = np.concatenate((grid_x.reshape(-1,1), grid_y.reshape(-1,1)), axis=1)
        return grid
    def addForaging(self, chosen_forage_indices, grid, simulator):
        for row in grid[chosen_forage_indices,:]:
            simulator.addForage(row[0],row[1])
        return simulator
    def addLarvaes(self, chosen_nest_indices, chosen_inner_nest_indices,grid, simulator):
        for row in grid[chosen_inner_nest_indices,:]:
            simulator.addAgent(self.generateLarvae(row[0],row[1],self.larvae_hunger_multiplier))
        for row in grid[chosen_nest_indices,:]:
            simulator.addAgent(self.generateLarvae(row[0],row[1]))
        return simulator
    def addWasps(self, chosen_nest_indices, chosen_inner_nest_indices, grid, simulator,path_finding):
        for _ in range(self.total_number_of_wasps): 
            if np.random.rand()<0.5:
                index = np.random.choice(chosen_nest_indices, 1, replace=False)
                simulator.addAgent(self.generateWasp(grid[index,0][0],grid[index,1][0],path_finding))
            else:
                index = np.random.choice(chosen_inner_nest_indices, 1, replace=False)
                simulator.addAgent(self.generateWasp(grid[index,0][0],grid[index,1][0],path_finding))
        return simulator
    def generateWasp(self,x,y,path_finding):
        wasp = Wasp(agent_id="W"+str(x)+str(y), x=x, y=y, food=1,hunger= 1,path_finding=path_finding)
        return wasp
    def generateSimulator(self,path_finding):
        simulator = Simulator()
        number_of_cells = np.random.randint(self.min_number_of_cells, self.max_number_of_cells+1)
        radius_nest = int(number_of_cells**(1/2))+1
        radius_inner_nest = int(radius_nest/2)
        radius_forage = radius_nest+self.forage_distance
        grid = self.generateGrid(radius_forage)

        inner_nest_indices = np.where(grid[:,0]**2+grid[:,1]**2<radius_inner_nest**2)[0]
        inner_chosen_nest_indices = np.random.choice(inner_nest_indices, int(self.nest_fill_percentage/2*inner_nest_indices.shape[0]), replace=False)

        nest_indices = np.where((grid[:,0]**2+grid[:,1]**2<radius_nest**2) & (grid[:,0]**2+grid[:,1]**2>radius_inner_nest**2))[0]
        chosen_nest_indices = np.random.choice(nest_indices, int(self.nest_fill_percentage*nest_indices.shape[0]), replace=False)

        forage_indices = np.where((grid[:,0]**2+grid[:,1]**2<radius_forage**2) & (grid[:,0]**2+grid[:,1]**2>radius_nest**2))[0]
        chosen_forage_indices = np.random.choice(forage_indices, int(self.forage_fill_percentage*forage_indices.shape[0]), replace=False)
        
        simulator = self.addForaging(chosen_forage_indices, grid, simulator)
        simulator = self.addLarvaes(chosen_nest_indices, inner_chosen_nest_indices,grid, simulator)
        simulator = self.addWasps(chosen_nest_indices, inner_chosen_nest_indices, grid, simulator, path_finding)
        
        return simulator
    
    def generateSimulationInstance(self,path_finding:str="greedy"):
        simulator = self.generateSimulator(path_finding)
        return simulator
