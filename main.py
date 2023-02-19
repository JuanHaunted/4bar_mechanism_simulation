import numpy as np
from math import radians, pi
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque



# We define all bar lengths and theta 5
L1 = 3
L2 = 2
L3 = 6
L4 = 5

L5 = 5
th_2_ini = 20
th_2_user = radians(th_2_ini) #Entry angle
th5 = radians(30)
w2 = 0.5
diff = 0.000001
disc = 150

# We calculate L6 using cosine law
L6 = np.sqrt((L5**2)+(L3**2)-(2*L5*L3*np.cos(radians(th5))))

# We define empty lists where we will store the positions of the revolutes
a = np.zeros((disc, 2), dtype=float) #r1
b = np.zeros((disc, 2), dtype=float) #r2
c = np.zeros((disc, 2), dtype=float) #r3
d = np.zeros((disc, 2), dtype=float) #r4
d[:, 0] = L1
p = np.zeros((disc, 2), dtype=float) #r5 - Punto a Gráficar
w4_w3 = np.zeros((disc, 2), dtype=float)
vpx_vpy = np.zeros((disc, 2), dtype=float)


# We create a list with L1, L2, L3, and L4
bars = [L1, L2, L3, L4]


def rotation_type(bar_list):
    global L1, L2, L3, L4
    # Length of longest and shortest bar
    l = max(bar_list)
    s = min(bar_list)

    # We remove l and s from the original bar_list to get p and q
    bar_list.remove(l)
    bar_list.remove(s)

    # Totales de s + l y p + q
    s_l = s + l
    p_q = sum(bar_list)

    # RI - Rotaciión Incompleta, RC - Rotacion completa, RCI = Caso Especial
    if s_l <= p_q:
        if L3 == s and s_l != p_q:
            return "RI" # Rotating Coupler case
        elif L2 == l and L4 == s and s_l != p_q:
            temp = L2
            L2 = L4
            L4 = temp
            return "RCI" # Cranck Rocker When L4 is Rocker
        elif L1 == L2 and L3 == L4:
            return "AD"
        #elif (L1 == l or L2 == l) and s_l == p_q:
            #return "ADH"
        else:
            return "RC"
    else:
        # ALl non-grashof are rotation incomplete
        if L2 == l or L1 == l:
            return "RING12"
        elif L3 == l or L4 == l:
            return "RING34"


def generate_angles(r_type):
    if r_type == "RC": #Grashof 1, Grashoff 2, Todos los no grashoff
        return np.linspace(th_2_user, th_2_user + (4*pi), disc)
    if r_type == "RI": # Caso Rotating Coupler
        th_max = np.arccos((L2**2+L1**2-(L3+L4)**2)/(2*L2*L1))
        th_min = np.arccos(((L2+L3)**2+L1**2-L4**2)/(2*(L2+L3)*L1))
        forward = np.linspace(th_min+diff, th_max- diff, disc//2)
        backward = np.linspace(th_max-diff, th_min + diff, disc//2)
        return np.concatenate([forward, backward])
    if r_type == "RCI":
        return np.linspace(th_2_user, th_2_user + (4*pi), disc)
    if r_type == "RING12":
        alpha = np.arccos((-(L3 + L4) **2 + L1 ** 2 + L2 ** 2) / (2 * L1 * L2))
        th_min = (2 * pi) - alpha
        th_max = (2 * pi) + alpha #Nos pasamos de 360 facherito
        forward = np.linspace(th_min + diff, th_max - diff, disc//2)
        backward = np.linspace(th_max - diff, th_min + diff, disc//2)
        return np.concatenate([forward, backward])
    if r_type == "RING34":
        alpha = np.arccos((-(L3 - L4) ** 2 + L1 ** 2 + L2 ** 2) / (2 * L1 * L2))
        th_min = alpha
        th_max = (2 * pi) - alpha  # Nos pasamos de 360 facherito
        forward = np.linspace(th_min + diff, th_max - diff, disc//2)
        backward = np.linspace(th_max - diff, th_min + diff, disc//2)
        return np.concatenate([forward, backward])
    if r_type == "AD": #Corregir
        return np.linspace(th_2_user, th_2_user + (4*pi), disc)
    if r_type == "ADH":
        th_max = np.arccos((L2 ** 2 + L1 ** 2 - (L3 + L4) ** 2) / (2 * L2 * L1))
        th_min = np.arccos(((L2 + L3) ** 2 + L1 ** 2 - L4 ** 2) / (2 * (L2 + L3) * L1))
        forward = np.linspace(th_min + diff, th_max - diff, disc // 2)
        backward = np.linspace(th_max - diff, th_min + diff, disc // 2)
        return np.concatenate([forward, backward])

def calculate_th3_th4_w3_w4(th2_angles):
    i = 0
    # We Calculate Bar Relation Coeficients for Later Use
    K1 = L1 / L2
    K2 = L1 / L4
    K3 = (L1 ** 2 + L2 ** 2 - L3 ** 2 + L4 ** 2) / (2 * L2 * L4)
    K4 = L1 / L3
    K5 = (-L1 ** 2 - L2 ** 2 - L3 ** 2 + L4 ** 2) / (2 * L2 * L3)
    for th2 in th2_angles:
        # We create an iterator to fill the revolute position arrays
        #Freudestein Equations
        A = np.cos(th2)*(1-K2)+K3-K1
        B = -2 * np.sin(th2)
        C = -np.cos(th2)*(1+K2)+K3+K1
        D = np.cos(th2)*(1+K4)+K5-K1
        E = -2 * np.sin(th2)
        F = np.cos(th2)*(K4-1)+K1+K5


        th3 = 2*np.arctan((-E-np.sqrt(E**2-(4*D*F)))/(2*D))
        th4 = 2*np.arctan((-B-np.sqrt(B**2-(4*A*C)))/(2*A))

        # We separate each relevant vector into its cordinates
        L2_X = L2*np.cos(th2)
        L2_Y = L2*np.sin(th2)
        L3_X = L3*np.cos(th3)
        L3_Y = L3*np.sin(th3)
        L4_X = L4*np.cos(th4)
        L4_Y = L4*np.sin(th4)
        L5_X = L5*np.cos(th3+th5)
        L5_Y = L5*np.sin(th3+th5)

        b[i, :] = np.array([L2_X, L2_Y])
        c[i, :] = np.array([L2_X+L3_X, L2_Y+L3_Y])
        p[i, :] = np.array([L2_X+L5_X, L2_Y+L5_Y])

        ################################################################################################################
        # Empieza el cálculo de velocidades, método x = A^-1 * B
        real_L4 = L4*(np.cos((pi/2)+th4))
        real_L3 = L3*(np.cos((pi/2)+th3+pi))
        imag_L4 = L4*(np.sin((pi/2)+th4))
        imag_L3 = L3*(np.sin((pi/2)+th3+pi))

        mat_A = np.array([[real_L4, real_L3] ,[imag_L4, imag_L3]])
        mat_B = np.array([[w2*L2*(np.cos((pi/2)+th2))],[w2*L2*(np.sin((pi/2)+th2))]])

        inv_mat_A = np.linalg.inv(mat_A)

        w4_w3[i, :] = (inv_mat_A @ mat_B).reshape((1, 2))

        w3 = w4_w3[i, 1]
        w4 = w4_w3[i, 0]

        vpx = w2*L2*(np.cos(th2+(pi/2))) + w3*L5*(np.cos(th2+th5+(pi/2)))
        vpy = w2*L2*(np.sin(th2+(pi/2))) + w3*L5*(np.sin(th2+th5+(pi/2)))

        vpx_vpy[i, :] = np.array([vpx, vpy])

        i += 1


#Función Nueva - Itera sobre un vector y remuve elimina los datos atípicos
def remove_atypical(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
           if x[i][j] > 50 or x[i][j]<-50:
               x[i][j] = np.NaN
    return x


rot_type = rotation_type(bars)
angles = generate_angles(rot_type)
calculate_th3_th4_w3_w4(angles)


all_coordinates = np.vstack([a, b, c, d, p])

print(rot_type)
print(w4_w3[:, 0])

# Nuevas - Elimina los datos atipicos en las velocidades
remove_atypical(vpx_vpy)
remove_atypical(w4_w3)



#Animation
max_xy = np.zeros((1, 2), dtype=float)
min_xy = np.zeros((1, 2), dtype=float)

max_xy[:, 0] = np.amax(all_coordinates[:, 0])
max_xy[:, 1] = np.amax(all_coordinates[:, 1])
min_xy[:, 0] = np.amin(all_coordinates[:, 0])
min_xy[:, 1] = np.amin(all_coordinates[:, 1])


fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal", autoscale_on=False, xlim=(min_xy[:, 0] - 1, max_xy[:, 0] + 1), ylim=(min_xy[:, 1] - 1, max_xy[:, 1] + 1))
#ax = fig.add_subplot(111, aspect="equal", autoscale_on=False, xlim=(-15, 15), ylim=(-15, 15)) #AD
history_len = 100
#Add Grid Lines, Titles and Labels
ax.grid(alpha=0.5)
ax.set_title("Motion Analysis")
ax.set_xticklabels([])
ax.set_yticklabels([])
(line, ) = ax.plot([], [], marker='o', lw=5, color="#8856a7")
(line2, ) = ax.plot([], [], marker='o', lw=5, color="#99d8c9")
(trace, ) = ax.plot([], [], '.-', lw=1, ms=2, color="orchid") ##0D006A
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

def init():
    line.set_data([], [])
    line2.set_data([], [])
    trace.set_data([], [])
    return line, line2, trace

def animate(i):
    thisx = [a[i, 0], b[i, 0], c[i, 0], d[i, 0]]
    thisy = [a[i, 1], b[i, 1], c[i, 1], d[i, 1]]
    line.set_data(thisx, thisy)
    thisx = [b[i, 0], p[i, 0], c[i, 0]]
    thisy = [b[i, 1], p[i, 1], c[i, 1]]
    line2.set_data(thisx, thisy)

    #Traza
    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[1])
    history_y.appendleft(thisy[1])

    trace.set_data(history_x, history_y)

    return line, line2, trace

ani = anim.FuncAnimation(fig, animate, init_func=init, frames=len(p), interval=30, blit=True, repeat=True, save_count=1500)

if rot_type == "RCI":
    plt.gca().invert_xaxis()

#Plots
fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
fig2.suptitle('Position and Velocity Analysis',fontsize=10)

ax1.plot(angles, w4_w3[:, 1], color="lime")
ax1.set_title("Angular Velocity in P (L3)", fontsize=8)
ax1.set_xlabel('Theta2', fontsize=7)
ax1.set_ylabel('w3', fontsize=7)
ax1.grid(True)

ax2.plot(p[:, 0], p[:, 1], color="mediumblue")
ax2.set_title("P point trayectory", fontsize=8)
ax2.set_xlabel('Px', fontsize=7)
ax2.set_ylabel('Py', fontsize=7)
ax2.grid(True)

ax3.plot(angles, vpx_vpy[:, 0], color="lightskyblue")
ax3.set_title("Linear Velocity in Px", fontsize=8)
ax3.set_xlabel('Theta2', fontsize=7)
ax3.set_ylabel('Vpx', fontsize=7)
ax3.grid(True)

ax4.plot(angles, vpx_vpy[:, 1], color="lightskyblue")
ax4.set_title("Linear Velocity in Py",fontsize=8)
ax4.set_xlabel('Theta2', fontsize=7)
ax4.set_ylabel('Vpy', fontsize=7)
ax4.grid(True)

ax5.plot(angles, p[:, 0], color="mediumblue")
ax5.set_title("Coordinates of Px",fontsize=8)
ax5.set_xlabel('Theta2', fontsize=7)
ax5.set_ylabel('Px', fontsize=7)
ax5.grid(True)

ax6.plot(angles, p[:, 1], color="mediumblue")
ax6.set_title("Coordinates of Py",fontsize=8)
ax6.set_xlabel('Theta2', fontsize=7)
ax6.set_ylabel('Py', fontsize=7)
ax6.grid(True)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.9)

plt.show()



