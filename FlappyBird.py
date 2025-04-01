import pygame
import os
import random
import neat

ai_playing = True  # Flag to indicate if AI is controlling the game
generation = 0  # Track the generation number

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 800

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))
BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird1.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird2.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird3.png'))),
]

pygame.font.init()
POINTS_FONT = pygame.font.SysFont('arial', 50)

class Bird:
    IMAGES = BIRD_IMAGES
    MAX_ROTATION = 25
    ROTATION_SPEED = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.height = self.y
        self.time = 0
        self.image_count = 0
        self.image = self.IMAGES[0]

    def jump(self):
        self.speed = -10.5
        self.time = 0
        self.height = self.y

    def move(self):
        # Calculate displacement
        self.time += 1
        displacement = 1.5 * (self.time ** 2) + self.speed * self.time

        # Limit displacement
        if displacement > 16:
            displacement = 16
        elif displacement < 0:
            displacement -= 2

        self.y += displacement

        # Adjust bird angle
        if displacement < 0 or self.y < (self.height + 50):
            if self.angle < self.MAX_ROTATION:
                self.angle = self.MAX_ROTATION
        else:
            if self.angle > -90:
                self.angle -= self.ROTATION_SPEED

    def draw(self, screen):
        # Select bird image based on animation frame
        self.image_count += 1

        if self.image_count < self.ANIMATION_TIME:
            self.image = self.IMAGES[0]
        elif self.image_count < self.ANIMATION_TIME * 2:
            self.image = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME * 3:
            self.image = self.IMAGES[2]
        elif self.image_count < self.ANIMATION_TIME * 4:
            self.image = self.IMAGES[1]
        elif self.image_count >= self.ANIMATION_TIME * 4 + 1:
            self.image = self.IMAGES[0]
            self.image_count = 0

        # Prevent wing flapping while falling
        if self.angle <= -80:
            self.image = self.IMAGES[1]
            self.image_count = self.ANIMATION_TIME * 2

        # Draw rotated bird image
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        image_center = self.image.get_rect(topleft=(self.x, self.y)).center
        rect = rotated_image.get_rect(center=image_center)
        screen.blit(rotated_image, rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    GAP = 200
    SPEED = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)
        self.PIPE_BOTTOM = PIPE_IMAGE
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.SPEED

    def draw(self, screen):
        screen.blit(self.PIPE_TOP, (self.x, self.top))
        screen.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        top_collision = bird_mask.overlap(top_mask, top_offset)
        bottom_collision = bird_mask.overlap(bottom_mask, bottom_offset)

        return top_collision or bottom_collision


class Ground:
    SPEED = 5
    WIDTH = GROUND_IMAGE.get_width()
    IMAGE = GROUND_IMAGE

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.SPEED
        self.x2 -= self.SPEED

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, screen):
        screen.blit(self.IMAGE, (self.x1, self.y))
        screen.blit(self.IMAGE, (self.x2, self.y))


def draw_screen(screen, birds, pipes, ground, score):
    screen.blit(BACKGROUND_IMAGE, (0, 0))
    for bird in birds:
        bird.draw(screen)
    for pipe in pipes:
        pipe.draw(screen)

    text = POINTS_FONT.render(f"Score: {score}", 1, (255, 255, 255))
    screen.blit(text, (SCREEN_WIDTH - 10 - text.get_width(), 10))

    if ai_playing:
        text = POINTS_FONT.render(f"Age: {generation}", 1, (255, 255, 255))
        screen.blit(text, (10, 10))

    ground.draw(screen)
    pygame.display.update()


def main(genomes, config):  # Fitness function
    global generation
    generation += 1

    networks = []
    genome_list = []
    birds = []

    # Initialize networks, genomes, and birds
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)
        genome.fitness = 0  # Initialize fitness
        genome_list.append(genome)
        birds.append(Bird(230, 350))

    ground = Ground(730)
    pipes = [Pipe(700)]
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    score = 0
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(30)

        # Check for user exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        # Determine which pipe to use for calculations
        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > (pipes[0].x + pipes[0].PIPE_TOP.get_width()):
                pipe_index = 1
        else:
            running = False
            break

        # Bird movement and fitness calculation
        for i, bird in enumerate(birds):
            bird.move()
            genome_list[i].fitness += 0.1  # Increment fitness over time

            # Check for jump based on neural network output
            output = networks[i].activate((bird.y,
                                           abs(bird.y - pipes[pipe_index].height),
                                           abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] > 0.5:
                bird.jump()

        ground.move()

        add_pipe = False
        pipes_to_remove = []
        for pipe in pipes:
            for i, bird in enumerate(birds):
                # Collision detection
                if pipe.collide(bird):
                    birds.pop(i)
                    genome_list.pop(i)
                    networks.pop(i)

                # Check if bird passed the pipe
                if not pipe.passed and bird.x > pipe.x:
                    pipe.passed = True
                    add_pipe = True

            pipe.move()

            # Remove pipes out of the screen
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes_to_remove.append(pipe)

        # Add a new pipe and increase fitness/score
        if add_pipe:
            score += 1
            pipes.append(Pipe(600))
            for genome in genome_list:
                genome.fitness += 5  # Reward for passing a pipe

        # Remove pipes marked for deletion
        for pipe in pipes_to_remove:
            pipes.remove(pipe)

        # Remove birds that hit the ground or fly too high
        for i, bird in enumerate(birds):
            if (bird.y + bird.image.get_height()) > ground.y or bird.y < 0:
                birds.pop(i)
                genome_list.pop(i)
                networks.pop(i)

        # Check if any bird reaches the fitness threshold
        for genome in genome_list:
            if genome.fitness >= 1000:
                print(f"Fitness threshold reached: {genome.fitness}")
                running = False  # Stop the simulation
                break

        draw_screen(screen, birds, pipes, ground, score)


def run(config_path):
    # Load NEAT configuration
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create NEAT population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Run the NEAT algorithm until a fitness threshold is reached or for 50 generations
    population.run(main, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
