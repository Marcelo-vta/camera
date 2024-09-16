import numpy as np
import itertools
import math

# Instalar a biblioteca cv2 pode ser um pouco demorado. Não deixe para ultima hora!
import cv2 as cv

def criar_indices(min_i, max_i, min_j, max_j):
    L = list(itertools.product(range(min_i, max_i), range(min_j, max_j)))
    idx_i = np.array([e[0] for e in L])
    idx_j = np.array([e[1] for e in L])
    idx = np.vstack((idx_i, idx_j))
    return idx


def run():
    # Essa função abre a câmera. Depois desta linha, a luz de câmera (se seu computador tiver) deve ligar.
    cap = cv.VideoCapture(0)

    # Aqui, defino a largura e a altura da imagem com a qual quero trabalhar.
    width = 320
    height = 240
    image = ""

    theta = 0.05

    sp_r = False
    sp_l = False
    mv_l = False
    mv_r = False
    mv_u = False
    mv_d = False
    reset = False

    rotation = 0
    x_translation = 0
    y_translation = 0

    # Dica: imagens menores precisam de menos processamento!!!

    

    # Talvez o programa não consiga abrir a câmera. Verifique se há outros dispositivos acessando sua câmera!
    if not cap.isOpened():
        print("Não consegui abrir a câmera!")
        exit()

    # Esse loop é igual a um loop de jogo: ele encerra quando apertamos 'q' no teclado.
    while True:
        # Captura um frame da câmera
        ret, frame = cap.read()

        if sp_r:
            rotation -= 1
        if sp_l:
            rotation += 1

        if mv_l:
            x_translation -= 2
        if mv_r:
            x_translation += 2

        if mv_u:
            y_translation -= 2
        if mv_d:
            y_translation += 2

        if reset:
            x_translation = 0
            y_translation = 0
            rotation = 0
            reset = False


        # A variável `ret` indica se conseguimos capturar um frame
        if not ret:
            print("Não consegui capturar frame!")
            break

        # Mudo o tamanho do meu frame para reduzir o processamento necessário
        # nas próximas etapas
        frame = cv.resize(frame, (width,height), interpolation =cv.INTER_AREA)

        # A variável image é um np.array com shape=(width, height, colors)
        image = np.array(frame).astype(float)/255
        image_ = np.zeros_like(image)

        To = np.array([[1,0,-(image.shape[0]/2)],[0,1,-(image.shape[1]/2)],[0,0,1]])
        Tc = np.linalg.inv(To)

        T = np.array([[1,0,y_translation],[0,1,x_translation],[0,0,1]])

        R = np.array([[math.cos(rotation * theta),-math.sin(rotation * theta),0],[math.sin(rotation * theta),math.cos(rotation * theta),0],[0,0,1]])

        B = T@Tc@R@To


        X = criar_indices(0, image.shape[0], 0, image.shape[1])
        X = np.vstack((X, np.ones( X.shape[1])))
        X = X.astype(int)

        X_ = np.linalg.inv(B)@X
        X_ = X_.astype(int)

        X_[0, :] = np.clip(X_[0, :], 0, image.shape[0] - 1)
        X_[1, :] = np.clip(X_[1, :], 0, image.shape[1] - 1)

        image_[X[0,:], X[1,:]] = image[X_[0,:], X_[1,:]]

        # Agora, mostrar a imagem na tela!
        cv.imshow('Minha Imagem!', image_)
        
        # Se aperto 'q', encerro o loop
        tecla = cv.waitKey(1)

        if tecla == ord('q'):
            return image
        if tecla == ord(','):
            sp_l = True
            sp_r = False
        if tecla == ord('.'):
            sp_r = True
            sp_l = False

        if tecla == ord('a'):
            mv_l = True
            mv_r = False
        if tecla == ord('d'):
            mv_r = True
            mv_l = False

        if tecla == ord('w'):
            mv_u = True
            mv_d = False
        if tecla == ord('s'):
            mv_u = False
            mv_d = True

        if tecla == ord('r'):
            sp_l = False
            sp_r = False
            mv_l = False
            mv_r = False
            mv_u = False
            mv_d = False
            reset = True
        if tecla == ord('p'):
            sp_l = False
            sp_r = False
            mv_l = False
            mv_r = False
            mv_u = False
            mv_d = False

        print(mv_u,)

        
            

    # Ao sair do loop, vamos devolver cuidadosamente os recursos ao sistema!
    cap.release()
    cv.destroyAllWindows()

print(run())

# for i in criar_indices(0,100,0,100)[0]:
#     print()

# print(len(criar_indices(0,100,0,100)[1]))
