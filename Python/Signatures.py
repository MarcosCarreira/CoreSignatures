# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Imports

# %%
import numpy as np
import pandas as pd

# %%
import matplotlib.pyplot as plt

# %%
# import matplotlib.animation
# import mpl_toolkits.mplot3d.art3d as art3d

# %%
import scipy, scipy.interpolate

# %%
# import sqlite3

# %%
import itertools

# %%
# from ipywidgets import interact
# import ipywidgets as widgets

# %%
# import IPython.display

# %%
# # %matplotlib inline

# %% [markdown]
# ## Basic 1d Path

# %% [markdown]
# ### Definition

# %%
f = lambda t: (t - 1)**3
t = np.linspace(0, 2, 200)

# %%
plt.figure(figsize=(5, 3))
plt.plot(t, f(t))
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('The 1-Dimensional Path $f(t) = (t-1)^3$')
plt.grid()
plt.show()

# %% [markdown]
# ### Interpolation

# %%
f = lambda t: (t - 1)**3
t = np.linspace(0, 2, 2000)
t_samples = np.linspace(0, 2, 8)

# %%
interpolation_methods = (('linear', 'Linear Interpolation'),
                         ('previous', 'Previous Point Interpolation'),
                         ('next', 'Next Point Interpolation'),
                         ('quadratic', 'Quadratic Interpolation'))

# %%
# figure_height = 8.5
figure_height = 6
fig, ax = plt.subplots(2, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                     sharex=True, sharey=True)
for method, ax in zip(interpolation_methods, ax.flatten()):
    f_interpolated = scipy.interpolate.interp1d(t_samples, f(t_samples), kind=method[0])

    p_time_series = ax.scatter(t_samples, f(t_samples), color='k')
    p_reconstructed, = ax.plot(t, f_interpolated(t))
    p_original, = ax.plot(t, f(t), linestyle=':')
    ax.set_xlabel('t')
    ax.set_ylabel('f(t)')
    ax.set_title('{}'.format(method[1]))
    ax.grid()
fig.legend([p_time_series, p_reconstructed, p_original],
           ['Data Stream', 'Reconstructed Path', 'Original Path'], loc=2)
plt.suptitle('Reconstructing a 1-D path From an Evenly Spaced Time Series')
plt.show()

# %%
f = lambda t: (t - 1)**3
t = np.linspace(0, 2, 2000)

# %%
np.random.seed(3)
t_samples = np.random.uniform(0, 2, size=8)
t_samples = np.insert(t_samples, [0, -1], [t[0], t[-1]])
reconstuctable_samples_mask = (t > min(t_samples)) & (t < max(t_samples))

# %%
interpolation_method = ('linear', 'Linear Interpolation')
f_interpolated = scipy.interpolate.interp1d(t_samples, f(t_samples), kind=interpolation_method[0])

# %%
plt.figure(figsize=(6, 4))
p_time_series = plt.scatter(t_samples, f(t_samples), color='k')
p_reconstructed, = plt.plot(t[reconstuctable_samples_mask], f_interpolated(t[reconstuctable_samples_mask]))
p_original, = plt.plot(t, f(t), linestyle=':')

plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('{}'.format(method[1]))
plt.legend([p_time_series, p_reconstructed, p_original], ['Data Stream', 'Reconstructed Path', 'Original Path'])
plt.title('Reconstructing a 1-D Path From an Evenly Spaced Time Series\nBased on Linear Interpolation')
plt.grid()
plt.show()


# %% [markdown]
# ## Computing Signatures

# %%
def compute_path_signature(X, a=0, b=1, steps=10**4, level_threshold=4):
    N = len(X)
    t = np.linspace(a, b, steps)
    dt = t[1] - t[0]
    X_t = [Xi(t) for Xi in X]
    t = t[:-1]
    dX_t = [np.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    
    signature = [[np.ones(len(t))]]
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(np.cumsum(previous_level_integral * dX_t[i]))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    
    return t, X_t, X_prime_t, signature, signature_terms


# %%
paths = ((lambda t: t, '$W_t = t$'),
         (lambda t: 2 * t, '$X_t = 2t$'),
         (lambda t: np.sin(np.pi / 2 * t) + 1, '$Y_t = sin(\\frac{\\pi}{2} t) + 1$'),
         (lambda t: np.where(t < 0.5, -1 * t, 3 * t - 2), '$Z_t = -t $ if $ t < 0.5; 3 * t - 2$ otherwise'))

# %%
paths2 = ((lambda t: (-1) ** (t), '$W_t = (-1) ** (t)$'),
         (lambda t: (-1) ** (t + 1), '$X_t = (-1) ** (t + 1)$'))

# %%
# ts, X_t, X_prime_t, signature, signature_terms = compute_path_signature([paths[2][0]], steps=10)

# %%
np.linspace(0, 9, 10)

# %%
ts, X_t, X_prime_t, signature, signature_terms = compute_path_signature([paths2[0][0]], a=0, b=9, steps=10)

# %%
ts

# %%
X_t

# %%
X_prime_t

# %%
signature

# %%
signature_terms

# %%
ts, X_t, X_prime_t, signature, signature_terms = compute_path_signature([paths2[1][0]], a=0, b=9, steps=10)

# %%
ts

# %%
X_t

# %%
X_prime_t

# %%
signature

# %%
signature_terms

# %%
paths3 = ((lambda t: (t) ** (10), '$W_t = (t) ** (10)$'),
         (lambda t: (t) ** (1 / 10), '$X_t = (t) ** (1 / 10)$'))

# %%
ts, X_t, X_prime_t, signature, signature_terms = compute_path_signature([paths3[0][0]], a=0, b=1, steps=20)

# %%
ts

# %%
X_t

# %%
X_prime_t

# %%
signature

# %%
signature_terms


# %%
def plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path_title):
    path_symbol = path_title[1]
    
    # Flatten lists
    signature = [integral for level in signature for integral in level] 
    signature_terms = [term for level in signature_terms for term in level]
    
    for i, ax in enumerate(axs):
        n = i + 1
        signature_filtered, signature_terms_filtered = zip(*[(integral, term)
                                                             for integral, term in zip(signature, signature_terms)
                                                             if len(term) > 0 and term[-1] == n])
        for integral, term in zip(signature_filtered, signature_terms_filtered):
            ax.plot(t, integral, label='$S(' + path_symbol + ')_{a,t}^{' +','.join([str(v) for v in term]) + '}$')
        
        ax.plot(t, X_prime_t[i], color='black', linestyle=':', label="$" + path_symbol + "'_t$" if len(axs) == 1 else "$" + path_symbol + "'_t^{}$".format(n))
        
        if len(axs) == 1:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_k}$'
            title = path_title
        else:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_{k-1}, nnn}$'.replace('nnn', str(n))
            title = 'Signature terms ' + ylabel
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('t', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
    
        ax.grid()
        ax.legend(fontsize=14)
        


# %%
figure_height = 9.5
plt.figure(figsize=(9,6))
plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path[1], [axs])


# %%
def plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path_title, axs):
    path_symbol = path_title[1]
    
    # Flatten lists
    signature = [integral for level in signature for integral in level] 
    signature_terms = [term for level in signature_terms for term in level]
    
    for i, ax in enumerate(axs):
        n = i + 1
        signature_filtered, signature_terms_filtered = zip(*[(integral, term)
                                                             for integral, term in zip(signature, signature_terms)
                                                             if len(term) > 0 and term[-1] == n])
        for integral, term in zip(signature_filtered, signature_terms_filtered):
            ax.plot(t, integral, label='$S(' + path_symbol + ')_{a,t}^{' +','.join([str(v) for v in term]) + '}$')
        
        ax.plot(t, X_prime_t[i], color='black', linestyle=':', label="$" + path_symbol + "'_t$" if len(axs) == 1 else "$" + path_symbol + "'_t^{}$".format(n))
        
        if len(axs) == 1:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_k}$'
            title = path_title
        else:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_{k-1}, nnn}$'.replace('nnn', str(n))
            title = 'Signature terms ' + ylabel
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('t', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
    
        ax.grid()
        ax.legend(fontsize=14)
        


# %%
figure_height = 9.5
fig, axs = plt.subplots(2, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                        sharex=True, sharey=True)

for path, ax in zip(paths, axs.flatten()):
    t, X_t, X_prime_t, signature, signature_terms = compute_path_signature([path[0]])
    plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path[1], [ax])

plt.suptitle('Examples of Signatures of 1-Dimensional Paths', fontsize=16)
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Basic 2d Path

# %%
# Read data corresponding to a single glyph entry, for a given user
conn = sqlite3.connect('data/corr_numeral_gesture_dataset/database.sqlite')
df = pd.read_sql_query("SELECT zglyph.zindex, zglyph.zsubject, zglyph.zcharacter, zglyph.zduration, "
                       "ztouch.ztimestamp, ztouch.zx, ztouch.zy "
                       "FROM zglyph, zstroke, ztouch "
                       "WHERE zstroke.zglyph=zglyph.z_pk AND ztouch.zstroke=zstroke.z_pk "
                       "AND zglyph.zsubject=1 AND zglyph.zcharacter=2;", conn)

df = df[df['ZINDEX'] == df['ZINDEX'].unique()[2]]
df = df[['ZTIMESTAMP', 'ZX', 'ZY']].set_index('ZTIMESTAMP', drop=True).sort_index()

# Process the data for playback
PLAYBACK_SPEED = 0.33
df.index = df.index / PLAYBACK_SPEED
FRAME_RATE=40
t = np.arange(df.index.min(), df.index.max(), 1/FRAME_RATE)
df = df.reindex(df.index.union(t)).interpolate().loc[t]

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 300)
ax.set_ylim(400, 0)
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_title('The 2-Dimensional Path $\mathbf{g}(t) = (x(t), y(t))$\n'
             'for a person drawing the digit "2" on a touchscreen')
ax.grid()
line_plot, = ax.plot([],[], alpha=0.15)
scatter_plot = ax.scatter([], [], color='k')
text_box = ax.text(250, 50, '', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Animate the plot
def animate(i):
    selected_data = df.iloc[:int(i+1)]
    line_plot.set_data(selected_data['ZX'], selected_data['ZY'])
    scatter_plot.set_offsets(selected_data[['ZX', 'ZY']].iloc[i])
    text_box.set_text('t={:.1f}s'.format(df.index[i]))
anim = matplotlib.animation.FuncAnimation(fig, animate, len(df), interval=1000/FRAME_RATE)

plt.close(fig)
IPython.display.HTML(anim.to_jshtml())


# %%
def plot_Y_against_X(a, b, X, Y, xlim, ylim, title):
    t = np.linspace(a, b, 200)
    X_t = X(t)
    Y_t = Y(t)
    dX_t = np.diff(X(t))
    dX_t = np.append(dX_t, dX_t[-1])
    
    plt.plot(X_t, Y_t, color='black')
    plt.fill_between(X_t, Y_t, where=~(Y_t >= 0) ^ (dX_t >= 0), facecolor='green', interpolate=True)
    plt.fill_between(X_t, Y_t, where=(Y_t >= 0) ^ (dX_t >= 0), facecolor='red', interpolate=True)
    
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('$X_t$')
    plt.ylabel('$Y_t$')
    plt.title(title)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

Slider = lambda value: widgets.SelectionSlider(options=[("{:.4}".format(i), i)
                                                        for i in sorted(np.append(np.linspace(0, np.pi*2), value))],
                                               continuous_update=False, value=value)

interact(lambda a, b: plot_Y_against_X(a, b,
                                       X=lambda t: t, Y=lambda t: np.cos(t),
                                       xlim=(0, np.pi*2), ylim=(-1.1, 1.1),
                                       title='Geometric interpretation of the path integral $\int_{a}^{b} Y_t \, dX_t$\n'
                                             'for $Y_t=cos(t)$, $X_t=t$ '
                                             'based on plotting $Y_t$ against $X_t$'),
         a=Slider(value=0),
         b=Slider(value=np.pi/2));

# %%
interact(lambda a, b: plot_Y_against_X(a, b,
                                       X=lambda t: np.sin(t), Y=lambda t: np.cos(t),
                                       xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
                                       title='Geometric interpretation of the path integral $\int_{a}^{b} Y_t \, dX_t$\n'
                                             'for $Y_t=cos(t)$, $X_t=sin(t)$ '
                                             'based on plotting $Y_t$ against $X_t$'),
         a=Slider(value=0),
         b=Slider(value=np.pi/2));


# %%
def plot_X_against_t(a, b, X, tlim, xlim, title):
    t = np.linspace(a, b, 200)
    X_t = X(t)
    
    plt.plot(t, X_t)
    plt.scatter([a, b], [X(a), X(b)], color='k')
    plt.hlines([X(a), X(b)], tlim[0], tlim[1], linestyle=':', color='gray')
    plt.vlines(tlim[1] - 0.25, X(a), X(b), linewidth=3, color=('green' if X(b) >= X(a) else 'red'))
    
    plt.xlim(*tlim)
    plt.ylim(*xlim)
    plt.xlabel('$t$')
    plt.ylabel('$X_t$')
    plt.title(title)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

interact(lambda a, b: plot_X_against_t(a, b,
                                       X=lambda t: np.sin(t),
                                       tlim=(0, np.pi*2 + 0.5),
                                       xlim=(-1.1, 1.1),
                                       title='Geometric interpretation of the path integral $\int_{a}^{b} Y_t \, dX_t$\n'
                                             'for $X_t=sin(t)$, $Y_t=1$ '
                                             'based on plotting $X_t$ against $t$'),
         a=Slider(value=0),
         b=Slider(value=np.pi/2));


# %%
def compute_path_signature(X, a=0, b=1, level_threshold=3):
    N = len(X)
    t = np.linspace(a, b, 10**4)
    dt = t[1] - t[0]
    X_t = [Xi(t) for Xi in X]
    t = t[:-1]
    dX_t = [np.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    
    signature = [[np.ones(len(t))]]
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(np.cumsum(previous_level_integral * dX_t[i]))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    
    return t, X_t, X_prime_t, signature, signature_terms

def plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path_title, axs):
    path_symbol = path_title[1]
    
    # Flatten lists
    signature = [integral for level in signature for integral in level] 
    signature_terms = [term for level in signature_terms for term in level]
    
    for i, ax in enumerate(axs):
        n = i + 1
        signature_filtered, signature_terms_filtered = zip(*[(integral, term)
                                                             for integral, term in zip(signature, signature_terms)
                                                             if len(term) > 0 and term[-1] == n])
        for integral, term in zip(signature_filtered, signature_terms_filtered):
            ax.plot(t, integral, label='$S(' + path_symbol + ')_{a,t}^{' +','.join([str(v) for v in term]) + '}$')
        
        ax.plot(t, X_prime_t[i], color='black', linestyle=':', label="$" + path_symbol + "'_t$" if len(axs) == 1 else "$" + path_symbol + "'_t^{}$".format(n))
        
        if len(axs) == 1:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_k}$'
            title = path_title
        else:
            ylabel = '$S(' + path_symbol + ')_{a,t}^{i_1, \ldots, i_{k-1}, nnn}$'.replace('nnn', str(n))
            title = 'Signature terms ' + ylabel
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('t', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
    
        ax.grid()
        ax.legend(fontsize=14)
        
paths = ((lambda t: t, '$W_t = t$'),
         (lambda t: 2 * t, '$X_t = 2t$'),
         (lambda t: np.sin(np.pi / 2 * t) + 1, '$Y_t = sin(\\frac{\\pi}{2} t) + 1$'),
         (lambda t: np.where(t < 0.5, -1 * t, 3 * t - 2), '$Z_t = -t $ if $ t < 0.5; 3 * t - 2$ otherwise'))

figure_height = 9.5
fig, axs = plt.subplots(2, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                        sharex=True, sharey=True)

for path, ax in zip(paths, axs.flatten()):
    t, X_t, X_prime_t, signature, signature_terms = compute_path_signature([path[0]])
    plot_path_signature(t, X_t, X_prime_t, signature, signature_terms, path[1], [ax])

plt.suptitle('Examples of Signatures of 1-Dimensional Paths', fontsize=16)
plt.show()

# %%
U = (((lambda t: np.where(t < 0.5, t, -3*t+2),
       lambda t: np.where(t < 0.5, 0.5*t, 0.25)), '$U_t'))

t = np.linspace(0, 1, 300)
U1_t = U[0][0](t)
U2_t = U[0][1](t)

figure_height = 6.0
plt.figure(figsize=(figure_height/3.5, figure_height))
plt.plot(U2_t, U1_t)
plt.xlabel('$U^2_t$')
plt.ylabel('$U^1_t$')
plt.title('The 2-Dimensional Path $U_t$')
plt.grid()
plt.show()

# %%
figure_height = 9.5
fig, axs = plt.subplots(1, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                        sharex=True, sharey=True)

t, U_t, U_prime_t, signature, signature_terms = compute_path_signature(U[0], a=0, b=1, level_threshold=3)
plot_path_signature(t, U_t, U_prime_t, signature, signature_terms, U[1], axs)

plt.suptitle('Signature of the 2-Dimensional Path U_t', fontsize=16)
plt.show()

# %%
print('Number of points in handwritten digit stream: {}'.format(len(df)))
print('Timestamp range: [{:.3f}, {:.3f}]'.format(min(df.index), max(df.index)))

# %%
W1_i = df['ZX']
W2_i = df['ZY']
t_i = df.index
plt.scatter(W1_i, W2_i)
plt.xlim(100, 300)
plt.ylim(350, 100)

for i, (x, y) in enumerate(zip(W1_i, W2_i)):
    if i == 0 or i == len(W1_i) - 1 or (i+1) % 10 == 0:
        plt.text(x, y, 'i={}'.format(i+1))

plt.xlabel('$\hat{W}^1[i]$ (horizontal coordinate)', fontsize=12)
plt.ylabel('$\hat{W}^2[i]$ (vertical coordinate)', fontsize=12)
plt.title('The 2-Dimensional Stream $\hat W[i]$\n(measurements for a person drawing the digit "2" on a touchscreen)')
plt.grid()
plt.show()

# %%
time_integrated_transformation = lambda t_i, W_i: np.vstack((t_i, W_i))

figure_height=8.0
fig = plt.figure(figsize=(figure_height * scipy.constants.golden_ratio, figure_height))
ax = fig.add_subplot(111, projection='3d')

def plot_stem3d(x, y, z, ax):
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):        
        line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1, 1))
        ax.add_line(line)

        if i == 0 or i == len(x) - 1 or (i+1) % 10 == 0:
            ax.text(xi, yi, zi, 'i={}'.format(i+1))

W_i = np.array(df[['ZX', 'ZY']]).T
t_i = df.index

T_time = time_integrated_transformation(t_i, W_i)
plot_stem3d(x=T_time[1],
            y=T_time[2],
            z=T_time[0],
            ax=ax)

ax.set_xlabel('$\hat{T}^2_{time}(\hat{W})[i]$ (horizontal coordinate)', fontsize=16)
ax.set_ylabel('$\hat{T}^3_{time}(\hat{W})[i]$ (vertical coordinate)', fontsize=16)
ax.set_zlabel('$\hat{T}^1_{time}(\hat{W})[i]$ (temporal coordinate)', fontsize=16)
plt.title('The Time-integrated Transformation of the 2-Dimensional Stream $\hat W[i]$\n(measurements for a person drawing the digit "2" on a touchscreen)',
          fontsize=16)
ax.set_xlim3d(100, 300)
ax.set_ylim3d(350, 100)
ax.set_zlim3d(min(t_i), max(t_i))

plt.grid()
plt.show()


# %%
def invisibility_reset_transformation(W_i):
    W_i = np.hstack((W_i, W_i[:, -1:]))
    W_i = np.hstack((W_i, np.zeros((2, 1))))
    invisibility_indicator = np.zeros(W_i.shape[1])
    invisibility_indicator[-2:] = 1
    
    return np.vstack((W_i, invisibility_indicator))

figure_height=8.0
fig = plt.figure(figsize=(figure_height * scipy.constants.golden_ratio, figure_height))
ax = fig.add_subplot(111, projection='3d')

W_i = np.array(df[['ZX', 'ZY']]).T
t_i = df.index

T_time = invisibility_reset_transformation(W_i)
plot_stem3d(x=T_time[0],
            y=T_time[1],
            z=T_time[2],
            ax=ax)

ax.set_xlabel('$\hat{T}^1_{time}(\hat{W})[i]$ (horizontal coordinate)', fontsize=16)
ax.set_ylabel('$\hat{T}^2_{time}(\hat{W})[i]$ (vertical coordinate)', fontsize=16)
ax.set_zlabel('$\hat{T}^3_{time}(\hat{W})[i]$ (invisibility coordinate)', fontsize=16)
plt.title('The Invisibility Reset Transformation of the 2-Dimensional Stream $\hat W$\n(measurements for a person drawing the digit "2" on a touchscreen)',
          fontsize=16)
ax.set_xlim3d(0, 300)
ax.set_ylim3d(350, 0)
ax.set_zlim3d(0, 1)

plt.grid()
plt.show()


# %%
def lead_lag_transformation(V_i):
    V_i = np.repeat(V_i, 2)
    
    return np.vstack((V_i[1:], V_i[:-1]))

V_i = [30, 25, 20, 40, 10]
i = range(1, len(V_i) + 1)

figure_height = 8.5
fig, axs = plt.subplots(1, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                        sharex=False, sharey=True)

axs[0].stem(i, V_i, use_line_collection=True)
axs[0].set_xticks(i)
axs[0].set_xlabel('i', fontsize=12)
axs[0].set_ylabel('$\hat{V}[i]$', fontsize=12)
axs[0].set_title('The 1-Dimensional Stream\n$\hat V = {}$'.format(V_i))
axs[0].grid()

T_leadlag = lead_lag_transformation(V_i)
axs[1].scatter(T_leadlag[0], T_leadlag[1], alpha=1)
axs[1].set_xlabel('$\hat{T}^1_{lead-lag}(\hat{V})[j]$', fontsize=12)
axs[1].set_ylabel('$\hat{T}^2_{lead-lag}(\hat{V})[j]$', fontsize=12)
axs[1].set_title('The Lead-Lag Transformation of the 1-Dimensional Stream\n$\hat V = {}$'.format(V_i))
axs[1].grid()

for i, (xi, yi) in enumerate(zip(T_leadlag[0], T_leadlag[1])):
    axs[1].text(xi, yi+0.5, 'j={}'.format(i+1))
        
plt.show()


# %%
def cumulative_sum_transformation(V_i):    
    return np.cumsum(V_i)

V_i = [30, 25, 20, 40, 10]
i = range(1, len(V_i) + 1)

figure_height = 8.5
fig, axs = plt.subplots(1, 2, figsize=(figure_height * scipy.constants.golden_ratio, figure_height),
                        sharex=False, sharey=True)

axs[0].stem(i, V_i, use_line_collection=True)
axs[0].set_xticks(i)
axs[0].set_xlabel('i', fontsize=12)
axs[0].set_ylabel('$\hat{V}[i]$', fontsize=12)
axs[0].set_title('The 1-Dimensional Stream\n$\hat V = {}$'.format(V_i))
axs[0].grid()

T_csum = cumulative_sum_transformation(V_i)
axs[1].stem(i, T_csum, use_line_collection=True)
axs[1].set_xticks(i)
axs[1].set_xlabel('i', fontsize=12)
axs[1].set_ylabel('$\hat{T}_{csum}(\hat{V})[i]$', fontsize=12)
axs[1].set_title('The Cumulative Sum Transformation of the 1-Dimensional Stream\n$\hat V = {}$'.format(V_i))
axs[1].grid()
        
plt.show()


# %%
def missing_data_transformation(W_i):
    missing_value_indices = np.array(range(0, W_i.shape[1], 10)[1:], dtype=int) - 1
    # Mark every 10th point as missing
    W_i[:, missing_value_indices] = np.nan
    # Replace missing observations with their predecessors
    W_i[:, missing_value_indices] = W_i[:, missing_value_indices - 1]
    missing_value_mask = np.zeros(W_i.shape[1])
    missing_value_mask[missing_value_indices] = 1
    
    return np.vstack((W_i, missing_value_mask))

figure_height=8.0
fig = plt.figure(figsize=(figure_height * scipy.constants.golden_ratio, figure_height))
ax = fig.add_subplot(111, projection='3d')

W_i = np.array(df[['ZX', 'ZY']]).T
t_i = df.index

T_missing = missing_data_transformation(W_i)
plot_stem3d(x=T_missing[0],
            y=T_missing[1],
            z=T_missing[2],
            ax=ax)

ax.set_xlabel('$\hat{T}^1_{time}(\hat{W})[i]$ (horizontal coordinate)', fontsize=16)
ax.set_ylabel('$\hat{T}^2_{time}(\hat{W})[i]$ (vertical coordinate)', fontsize=16)
ax.set_zlabel('$\hat{T}^3_{time}(\hat{W})[i]$ (missing data coordinate)', fontsize=16)
plt.title('The Missing Data Transformation of the 2-Dimensional Stream $\hat W$\n(measurements for a person drawing the digit "2" on a touchscreen)',
          fontsize=16)
ax.set_xlim3d(0, 300)
ax.set_ylim3d(350, 0)
ax.set_zlim3d(0, 1)

plt.grid()
plt.show()

# %%
