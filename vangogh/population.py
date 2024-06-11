import numpy as np
from PIL import Image
from vangogh.HomogenityProcessing import homogeneous_mask

IMAGE_PATH = "./img/reference_image_resized.jpg"
GUIDED_POINTS_RATIO = 0.25
THRESHOLD = 2
KERNEL_SIZE = 4

class Population:
    def __init__(self, population_size, genotype_length, initialization, ratio = GUIDED_POINTS_RATIO):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))
        self.initialization = initialization
        self.ratio = ratio

    def initialize(self, feature_intervals):
        n = self.genes.shape[0]
        l = self.genes.shape[1]

        if self.initialization == "RANDOM":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i

        elif self.initialization == "GUIDED":
            image = Image.open(IMAGE_PATH)
            image = image.convert("RGB")
            image_np = np.array(image)

            mask = homogeneous_mask(IMAGE_PATH, THRESHOLD, KERNEL_SIZE)
            sampling_array = np.where(mask == 1)

            length = len(sampling_array)
            num_points = l / 5

            for i in range(int(num_points * self.ratio)):
                indices = np.random.randint(low=0, high=length, size=n)
                guided_points_x = sampling_array[0][indices]
                guided_points_y = sampling_array[1][indices]

                r = image_np[guided_points_x, guided_points_y, 0]
                g = image_np[guided_points_x, guided_points_y, 1]
                b = image_np[guided_points_x, guided_points_y, 2]

                #print(len(guided_points_x), len(guided_points_y), len(r), len(g), len(b))

                self.genes[:, i * 5] = guided_points_x
                self.genes[:, i * 5 + 1] = guided_points_y
                self.genes[:, i * 5 + 2] = r
                self.genes[:, i * 5 + 3] = g
                self.genes[:, i * 5 + 4] = b         
            
            for i in range(int(num_points * self.ratio) * 5, l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i

        elif self.initialization == "GUIDED_V2":
            image = Image.open(IMAGE_PATH)
            image = image.convert("RGB")
            image_np = np.array(image)

            mask = homogeneous_mask(IMAGE_PATH, THRESHOLD, KERNEL_SIZE)
            sampling_array = np.where(mask == 1)

            length = len(sampling_array[0])
            num_points = l / 5

            width = image_np.shape[1]
            height = image_np.shape[0],


            for i in range(int(num_points)):
                if i < int(num_points * self.ratio):
                    indices = np.random.randint(low=0, high=length, size=n)
                    
                    guided_points_x = sampling_array[0][indices]
                    guided_points_y = sampling_array[1][indices]
                    
                    # print(guided_points_x, guided_points_y)
                    r = image_np[guided_points_x, guided_points_y, 0]
                    g = image_np[guided_points_x, guided_points_y, 1]
                    b = image_np[guided_points_x, guided_points_y, 2]

                    #print(len(guided_points_x), len(guided_points_y), len(r), len(g), len(b))

                    self.genes[:, i * 5] = guided_points_x
                    self.genes[:, i * 5 + 1] = guided_points_y
                    self.genes[:, i * 5 + 2] = r
                    self.genes[:, i * 5 + 3] = g
                    self.genes[:, i * 5 + 4] = b         
                else:
                    index_0 = np.random.randint(low=0, high= height, size=n)
                    index_1 = np.random.randint(low=0, high= width, size=n)
                    
                    r = image_np[index_0, index_1, 0]
                    g = image_np[index_0, index_1, 1]
                    b = image_np[index_0, index_1, 2]
                    print(list(zip(r, g, b)))
                    #print(len(guided_points_x), len(guided_points_y), len(r), len(g), len(b))

                    self.genes[:, i * 5] = index_0
                    self.genes[:, i * 5 + 1] = index_1
                    self.genes[:, i * 5 + 2] = r
                    self.genes[:, i * 5 + 3] = g
                    self.genes[:, i * 5 + 4] = b     

                
        else:
            raise Exception("Unknown initialization method")

    def stack(self, other):
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self):
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices):
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)
