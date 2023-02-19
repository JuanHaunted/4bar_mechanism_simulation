import numpy as np



a = np.array([1, -1, 2, 3])
b = np.array([4, 5, -4, 3])

print(np.dot(a, b))

'''
def generate_angles(r_type):
    if r_type == "RC": #Grashof 1, Grashoff 2, Todos los no grashoff
        return np.linspace(th_2_user, th_2_user + (4*pi), 100)
    if r_type == "RI": # Caso Rotating Coupler
        th_max = np.arccos((L2**2+L1**2-(L3+L4)**2)/(2*L2*L1))
        th_min = np.arccos(((L2+L3)**2+L1**2-L4**2)/(2*(L2+L3)*L1))
        forward = np.linspace(th_min+diff, th_max- diff, 50)
        backward = np.linspace(th_max-diff, th_min + diff, 50)
        return np.concatenate([forward, backward])
    if r_type == "RCI":
        #COmentar Luego sapo
        th_max = np.arccos(((L3 + L4) ** 2 - L1 ** 2 - L2 ** 2) / (-2 * L1 * L2))
        th_min = np.arccos(((L3 - L4) ** 2 - L1 ** 2 - L2 ** 2) / (-2 * L1 * L2))
        print(th_max)
        print(th_min)
        forward = np.linspace(th_min - diff, th_max + diff, 50)
        backward = np.linspace(th_max + diff, th_min - diff, 50)
        
        return np.linspace(th_2_user, th_2_user + (4*pi), 100)
    if r_type == "RING12":
        alpha = np.arccos((-(L3 + L4) **2 + L1 ** 2 + L2 ** 2) / (2 * L1 * L2))
        th_min = (2 * pi) - alpha
        th_max = (2 * pi) + alpha #Nos pasamos de 360 facherito
        forward = np.linspace(th_min + diff, th_max - diff, 50)
        backward = np.linspace(th_max - diff, th_min + diff, 50)
        return np.concatenate([forward, backward])
    if r_type == "RING34":
        alpha = np.arccos((-(L3 - L4) ** 2 + L1 ** 2 + L2 ** 2) / (2 * L1 * L2))
        th_min = alpha
        th_max = (2 * pi) - alpha  # Nos pasamos de 360 facherito
        forward = np.linspace(th_min + diff, th_max - diff, 50)
        backward = np.linspace(th_max - diff, th_min + diff, 50)
        return np.concatenate([forward, backward])
    if r_type == "AD": #Corregir
        return np.linspace(th_2_user, th_2_user + (4*pi), 100)
'''