import pygame

x, y, length, width = 284, 250, 68, 250
pygame.init()
GUI = pygame.display.set_mode((800,600))
pygame.display.set_caption("The incredible guessing game")
run = True
while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
	pygame.draw.rect(GUI, (255,210,0), (x,y,length,width))
	pygame.display.update()
pygame.quit()
