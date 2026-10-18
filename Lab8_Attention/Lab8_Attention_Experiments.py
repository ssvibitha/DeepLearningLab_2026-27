# %% [markdown]
# # Lab 8 & 9 — Effect of Attention Type and Attention Position in a Pretrained ResNet50
# **Dataset:** TensorFlow Flowers (5 classes) | **Backbone:** ResNet50 (ImageNet pretrained)

# %% [markdown]
# ## Cell 1 — Install & Imports

# %%
# !pip install tensorflow-datasets -q

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (confusion_matrix,
                              precision_score, recall_score, f1_score)
import seaborn as sns
import pandas as pd

tf.random.set_seed(42)
np.random.seed(42)

# Plot settings
mpl.rcParams['font.family'] = 'Liberation Serif'
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['savefig.dpi'] = 600

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# %% [markdown]
# ## Cell 2 — Dataset: TF Flowers (70/15/15 split)

# %%
IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
NUM_CLASSES = 5

ds_full, info = tfds.load('tf_flowers', split='train', with_info=True,
                           as_supervised=True, shuffle_files=False)
total = info.splits['train'].num_examples   # 3670

def get_split(ds, total, train_frac=0.70, val_frac=0.15):
    n_train = int(total * train_frac)
    n_val   = int(total * val_frac)
    ds = ds.shuffle(total, seed=42, reshuffle_each_iteration=False)
    return ds.take(n_train), ds.skip(n_train).take(n_val), ds.skip(n_train + n_val)

train_raw, val_raw, test_raw = get_split(ds_full, total)
print(f"Train: ~{int(total*0.70)} | Val: ~{int(total*0.15)} | Test: ~{total-int(total*0.70)-int(total*0.15)}")

# %% [markdown]
# ## Cell 3 — Preprocessing & Augmentation

# %%
preprocess_fn = tf.keras.applications.resnet50.preprocess_input

aug_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name='augmentation')

def preprocess_train(img, label):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = aug_layer(img, training=True)
    img = preprocess_fn(img)
    return img, label

def preprocess_eval(img, label):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_raw.map(preprocess_train, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (val_raw.map(preprocess_eval,   num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
test_ds  = (test_raw.map(preprocess_eval,  num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# %% [markdown]
# ## Cell 4 — Sample Images from All 5 Classes

# %%
class_imgs = {i: None for i in range(NUM_CLASSES)}
for img, lbl in test_raw:
    l = int(lbl.numpy())
    if class_imgs[l] is None:
        class_imgs[l] = tf.image.resize(img, [224, 224]).numpy().astype(np.uint8)
    if all(v is not None for v in class_imgs.values()):
        break

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, ax in enumerate(axes):
    ax.imshow(class_imgs[i])
    ax.set_title(CLASS_NAMES[i])
    ax.axis('off')
plt.suptitle("Sample Images — TF Flowers Dataset", fontweight='bold')
plt.tight_layout()
plt.savefig("sample_images.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 5 — Attention Module Definitions

# %%
# ── Channel Attention (SE-style) ─────────────────────────────────────────────
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, r=16, **kwargs):
        super().__init__(**kwargs)
        self.gap     = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1     = tf.keras.layers.Dense(max(channels // r, 1), activation='relu')
        self.fc2     = tf.keras.layers.Dense(channels, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))

    def call(self, x):
        z = self.gap(x)               # (B, C)
        m = self.fc2(self.fc1(z))     # (B, C)
        m = self.reshape(m)           # (B,1,1,C)
        return x * m, m               # refined features, weights

# ── Spatial Attention ─────────────────────────────────────────────────────────
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, kernel_size, padding='same',
                                           activation='sigmoid')

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B,H,W,1)
        mx  = tf.reduce_max(x,  axis=-1, keepdims=True)  # (B,H,W,1)
        m   = self.conv(tf.concat([avg, mx], axis=-1))   # (B,H,W,1)
        return x * m, m

# ── Lightweight Self-Attention ─────────────────────────────────────────────────
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, d=64, **kwargs):
        super().__init__(**kwargs)
        self.d     = d
        self.Wq    = tf.keras.layers.Dense(d)
        self.Wk    = tf.keras.layers.Dense(d)
        self.Wv    = tf.keras.layers.Dense(channels)
        self.gamma = tf.Variable(0.0, trainable=True, name='gamma')

    def call(self, x):
        B, H, W, C = tf.unstack(tf.shape(x))
        flat = tf.reshape(x, [B, H * W, C])
        A = tf.nn.softmax(
            tf.matmul(self.Wq(flat), self.Wk(flat), transpose_b=True)
            / tf.math.sqrt(tf.cast(self.d, tf.float32)), axis=-1)
        Y = tf.reshape(tf.matmul(A, self.Wv(flat)), [B, H, W, C])
        return x + self.gamma * Y, A

# ── ECA (Efficient Channel Attention) ─────────────────────────────────────────
class ECA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gap  = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=3, padding='same',
                                           use_bias=False)

    def call(self, x):
        z = tf.expand_dims(self.gap(x), axis=-1)          # (B,C,1)
        m = tf.reshape(tf.nn.sigmoid(self.conv(z)),
                       [-1, 1, 1, tf.shape(x)[-1]])        # (B,1,1,C)
        return x * m, m

# ── CBAM ──────────────────────────────────────────────────────────────────────
class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, r=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ca = ChannelAttention(channels, r)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        x, cw = self.ca(x)
        x, sm = self.sa(x)
        return x, cw, sm

# %% [markdown]
# ## Cell 6 — Model Builder

# %%
# Stage output layer names in ResNet50
STAGE_LAYER = {
    'P2': 'conv2_block3_out',   # 56×56×256
    'P3': 'conv3_block4_out',   # 28×28×512
    'P4': 'conv4_block6_out',   # 14×14×1024
}
STAGE_CHANNELS = {'P2': 256, 'P3': 512, 'P4': 1024}

def _apply_attn(x, attn_type, C, kernel_size, order):
    """Insert attention block and return refined tensor."""
    if attn_type is None:
        return x
    if attn_type in ('channel', 'se'):
        x, _ = ChannelAttention(C)(x)
    elif attn_type == 'spatial':
        x, _ = SpatialAttention(kernel_size)(x)
    elif attn_type == 'channel_spatial':
        if order == 'cs':
            x, _ = ChannelAttention(C)(x)
            x, _ = SpatialAttention(kernel_size)(x)
        else:
            x, _ = SpatialAttention(kernel_size)(x)
            x, _ = ChannelAttention(C)(x)
    elif attn_type == 'self':
        x, _ = SelfAttention(C)(x)
    elif attn_type == 'combined':
        x, _ = ChannelAttention(C)(x)
        x, _ = SpatialAttention(kernel_size)(x)
        x, _ = SelfAttention(C)(x)
    elif attn_type == 'eca':
        x, _ = ECA()(x)
    elif attn_type == 'cbam':
        x, _, _ = CBAM(C, kernel_size=kernel_size)(x)
    return x

def build_model(attn_type=None, position='P3', kernel_size=7,
                order='cs', multi_pos=None):
    """
    attn_type  : None | 'channel' | 'spatial' | 'channel_spatial' |
                 'self' | 'combined' | 'se' | 'eca' | 'cbam'
    position   : 'P2' | 'P3' | 'P4'  (used when multi_pos is None)
    multi_pos  : list of positions, e.g. ['P3', 'P4']
    """
    base = tf.keras.applications.ResNet50(weights='imagenet',
                                          include_top=False,
                                          input_shape=(224, 224, 3))
    positions = multi_pos if multi_pos else [position]

    # Get layer indices for insertion points
    layer_names = [l.name for l in base.layers]
    insert_at = sorted([(p, layer_names.index(STAGE_LAYER[p]))
                        for p in positions], key=lambda kv: kv[1])

    # Rebuild graph with attention inserted
    x   = base.layers[0].output
    x   = base.layers[1](x)
    ptr = 2
    for pos, end_idx in insert_at:
        while ptr <= end_idx:
            x = base.layers[ptr](x)
            ptr += 1
        x = _apply_attn(x, attn_type, STAGE_CHANNELS[pos], kernel_size, order)

    while ptr < len(base.layers):
        x = base.layers[ptr](x)
        ptr += 1

    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)

# %% [markdown]
# ## Cell 7 — Training Helper

# %%
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 5
LR_HEAD    = 1e-3
LR_FINETUNE = 1e-5

def compile_and_train(model, name):
    """Two-stage training: freeze → train head; unfreeze conv4+conv5 → fine-tune."""
    # Stage 1: freeze ResNet50 backbone
    base = model.layers[0]
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_HEAD),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_FROZEN, verbose=1)

    # Stage 2: unfreeze last two conv stages
    for layer in base.layers:
        if 'conv4' in layer.name or 'conv5' in layer.name:
            layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_FINETUNE, verbose=1)

    # Merge histories
    return model, {k: h1.history[k] + h2.history[k] for k in h1.history}

def evaluate_model(model, name):
    y_true, y_pred = [], []
    for imgs, lbls in test_ds:
        preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
        y_pred.extend(preds); y_true.extend(lbls.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc  = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true,    y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true,        y_pred, average='macro', zero_division=0)

    # Inference time
    sample_batch = next(iter(test_ds.take(1)))[0]
    t0 = time.time()
    model.predict(sample_batch, verbose=0)
    inf_ms = (time.time() - t0) / len(sample_batch) * 1000

    n_params = model.count_params()
    print(f"\n{'='*55}\n{name}\n"
          f"Acc:{acc:.4f}  P:{prec:.4f}  R:{rec:.4f}  F1:{f1:.4f}\n"
          f"Params:{n_params:,}  Inf:{inf_ms:.2f} ms/img\n{'='*55}")

    return dict(name=name, acc=acc, prec=prec, rec=rec, f1=f1,
                params=n_params, inf_time=inf_ms,
                y_true=y_true, y_pred=y_pred)

all_results = {}   # stores all experiment results

# %% [markdown]
# ## Cell 8 — A0: Baseline ResNet50

# %%
model_A0 = build_model(attn_type=None)
model_A0, hist_A0 = compile_and_train(model_A0, 'A0_Baseline')
res_A0 = evaluate_model(model_A0, 'A0: Baseline')
all_results['A0'] = {**res_A0, 'history': hist_A0}

# %% [markdown]
# ## Cell 9 — A1: Channel Attention @ P3

# %%
model_A1 = build_model(attn_type='channel', position='P3')
model_A1, hist_A1 = compile_and_train(model_A1, 'A1_Channel')
res_A1 = evaluate_model(model_A1, 'A1: Channel@P3')
all_results['A1'] = {**res_A1, 'history': hist_A1}

# %% [markdown]
# ## Cell 10 — A2: Spatial Attention @ P3

# %%
model_A2 = build_model(attn_type='spatial', position='P3', kernel_size=7)
model_A2, hist_A2 = compile_and_train(model_A2, 'A2_Spatial')
res_A2 = evaluate_model(model_A2, 'A2: Spatial@P3')
all_results['A2'] = {**res_A2, 'history': hist_A2}

# %% [markdown]
# ## Cell 11 — A3: Channel+Spatial @ P3

# %%
model_A3 = build_model(attn_type='channel_spatial', position='P3', order='cs')
model_A3, hist_A3 = compile_and_train(model_A3, 'A3_ChanSp')
res_A3 = evaluate_model(model_A3, 'A3: Chan+Spatial@P3')
all_results['A3'] = {**res_A3, 'history': hist_A3}

# %% [markdown]
# ## Cell 12 — A4: Self-Attention @ P3

# %%
model_A4 = build_model(attn_type='self', position='P3')
model_A4, hist_A4 = compile_and_train(model_A4, 'A4_Self')
res_A4 = evaluate_model(model_A4, 'A4: Self-Attn@P3')
all_results['A4'] = {**res_A4, 'history': hist_A4}

# %% [markdown]
# ## Cell 13 — A5: Combined (Chan+Spatial+Self) @ P3

# %%
model_A5 = build_model(attn_type='combined', position='P3')
model_A5, hist_A5 = compile_and_train(model_A5, 'A5_Combined')
res_A5 = evaluate_model(model_A5, 'A5: Combined@P3')
all_results['A5'] = {**res_A5, 'history': hist_A5}

# %% [markdown]
# ## Cell 14 — Plot: Training & Validation Accuracy/Loss (Exp A — Attention Type)

# %%
exp_A_hist = {
    'A0: Baseline':  hist_A0,
    'A1: Channel':   hist_A1,
    'A2: Spatial':   hist_A2,
    'A3: Chan+Sp':   hist_A3,
    'A4: Self-Attn': hist_A4,
}
colors = plt.cm.tab10(np.linspace(0, 0.9, len(exp_A_hist)))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for (nm, h), c in zip(exp_A_hist.items(), colors):
    ep = range(1, len(h['accuracy']) + 1)
    axes[0].plot(ep, h['accuracy'],     color=c, label=nm)
    axes[0].plot(ep, h['val_accuracy'], color=c, linestyle='--')
    axes[1].plot(ep, h['loss'],         color=c, label=nm)
    axes[1].plot(ep, h['val_loss'],     color=c, linestyle='--')

for ax, ylabel, title in zip(axes,
    ['Accuracy', 'Loss'],
    ['Training (—) & Validation (--) Accuracy',
     'Training (—) & Validation (--) Loss']):
    ax.axvline(EPOCHS_FROZEN, color='gray', linestyle=':', alpha=0.7, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='best')

plt.suptitle('Experiment A — Attention Type Ablation @ P3', fontweight='bold')
plt.tight_layout()
plt.savefig("exp_A_train_curves.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 15 — Exp B: Attention-Position Ablation (Channel+Spatial at P2, P3, P4)

# %%
# B1: P2
model_B1 = build_model(attn_type='channel_spatial', position='P2', order='cs')
model_B1, hist_B1 = compile_and_train(model_B1, 'B1_P2')
res_B1 = evaluate_model(model_B1, 'B1: CS@P2')
all_results['B1'] = {**res_B1, 'history': hist_B1}

# B2: P3 — reuse A3
res_B2 = {**res_A3}; res_B2['name'] = 'B2: CS@P3'
all_results['B2'] = {**res_B2, 'history': hist_A3}

# B3: P4
model_B3 = build_model(attn_type='channel_spatial', position='P4', order='cs')
model_B3, hist_B3 = compile_and_train(model_B3, 'B3_P4')
res_B3 = evaluate_model(model_B3, 'B3: CS@P4')
all_results['B3'] = {**res_B3, 'history': hist_B3}

# %% [markdown]
# ## Cell 16 — Plot: Attention Position vs Accuracy (Bar)

# %%
pos_labels = ['P2\n(56×56, 256ch)', 'P3\n(28×28, 512ch)', 'P4\n(7×7, 2048ch)']
pos_accs   = [res_B1['acc'], res_A3['acc'], res_B3['acc']]
pos_f1s    = [res_B1['f1'],  res_A3['f1'],  res_B3['f1']]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bar_colors = ['#4e9af1', '#f18f4e', '#4ef1a0']
for ax, vals, ylabel, title in zip(axes,
    [pos_accs, pos_f1s],
    ['Test Accuracy', 'Macro F1'],
    ['Attention Position vs Accuracy', 'Attention Position vs Macro F1']):
    bars = ax.bar(pos_labels, vals, color=bar_colors, width=0.5, label='Models')
    ax.axhline(res_A0['acc'], color='red', linestyle='--', label=f"Baseline ({res_A0['acc']:.3f})")
    ax.set_xlabel('Attention Position'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    ax.set_ylim(min(vals) - 0.06, 1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f'{v:.4f}', ha='center', fontweight='bold')

plt.suptitle('Experiment B — Attention Position Ablation (Channel+Spatial)', fontweight='bold')
plt.tight_layout()
plt.savefig("exp_B_position.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 17 — Exp C: Multiple Attention Positions (P3+P4, P2+P3+P4)

# %%
# P3+P4
model_C1 = build_model(attn_type='channel_spatial', multi_pos=['P3', 'P4'])
model_C1, hist_C1 = compile_and_train(model_C1, 'C1_P3P4')
res_C1 = evaluate_model(model_C1, 'C1: CS@P3+P4')
all_results['C1'] = {**res_C1, 'history': hist_C1}

# P2+P3+P4
model_C2 = build_model(attn_type='channel_spatial', multi_pos=['P2', 'P3', 'P4'])
model_C2, hist_C2 = compile_and_train(model_C2, 'C2_P2P3P4')
res_C2 = evaluate_model(model_C2, 'C2: CS@P2+P3+P4')
all_results['C2'] = {**res_C2, 'history': hist_C2}

# %% [markdown]
# ## Cell 18 — Plot: Multi-Position Accuracy & Parameter Count

# %%
mp_labels = ['P3\nonly', 'P4\nonly', 'P3+P4', 'P2+P3+P4']
mp_accs   = [res_A3['acc'],  res_B3['acc'],  res_C1['acc'],  res_C2['acc']]
mp_params = [res_A3['params'],res_B3['params'],res_C1['params'],res_C2['params']]

fig, ax1 = plt.subplots(figsize=(11, 6))
x = np.arange(len(mp_labels))
bars = ax1.bar(x, mp_accs, color='steelblue', alpha=0.85, label='Multi-Position Acc')
ax1.set_xticks(x); ax1.set_xticklabels(mp_labels)
ax1.set_xlabel('Stage Placement Configuration')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Experiment C — Multi-Position Attention Comparison')
ax1.axhline(res_A0['acc'], color='red', linestyle='--', label=f"Baseline ({res_A0['acc']:.3f})")
for i, (a, p) in enumerate(zip(mp_accs, mp_params)):
    ax1.text(i, a + 0.003, f'{a:.3f}\n({p/1e6:.1f}M)',
             ha='center', fontsize=12, fontweight='bold')
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
ax1.legend(loc='lower right')
plt.tight_layout()
plt.savefig("exp_C_multi_position.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 19 — Exp D: Attention Ordering (Channel→Spatial vs Spatial→Channel)

# %%
# CS order → already A3; SC order:
model_D1 = build_model(attn_type='channel_spatial', position='P3', order='sc')
model_D1, hist_D1 = compile_and_train(model_D1, 'D1_SC')
res_D1 = evaluate_model(model_D1, 'D1: Spatial→Channel@P3')
all_results['D1'] = {**res_D1, 'history': hist_D1}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, key, ylabel in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
    for h, nm, c in zip([hist_A3, hist_D1],
                         ['Chan→Spatial (CS)', 'Spatial→Chan (SC)'],
                         ['royalblue', 'tomato']):
        ep = range(1, len(h[key]) + 1)
        ax.plot(ep, h[key],          color=c, label=f'{nm} Train')
        ax.plot(ep, h['val_' + key], color=c, linestyle='--', label=f'{nm} Val')
    ax.axvline(EPOCHS_FROZEN, color='gray', linestyle=':', alpha=0.7, label='Unfreeze')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_title(f'Ordering — {ylabel}')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='best')

plt.suptitle('Experiment D — Attention Ordering at P3', fontweight='bold')
plt.tight_layout()
plt.savefig("exp_D_order.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 20 — Exp E: Spatial Attention Kernel Size (k = 3, 5, 7)

# %%
kernel_results = {}
for k in [3, 5, 7]:
    m = build_model(attn_type='spatial', position='P3', kernel_size=k)
    m, h = compile_and_train(m, f'E_k{k}')
    r = evaluate_model(m, f'E: Spatial k={k}@P3')
    kernel_results[k] = {**r, 'history': h}
    all_results[f'E_k{k}'] = kernel_results[k]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in zip(axes,
    ['acc', 'f1', 'inf_time'],
    ['Test Accuracy', 'Macro F1', 'Inference Time (ms/img)']):
    vals = [kernel_results[k][metric] for k in [3, 5, 7]]
    bars = ax.bar(['k=3', 'k=5', 'k=7'], vals,
                  color=['#5b9bd5', '#ed7d31', '#70ad47'], label='Kernel Models')
    ax.set_xlabel('Kernel Size'); ax.set_ylabel(title); ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.002,
                f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Experiment E — Spatial Attention Kernel Size', fontweight='bold')
plt.tight_layout()
plt.savefig("exp_E_kernel.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 21 — Additional: ECA and CBAM @ P3

# %%
# SE ≈ Channel Attention (A1), already done

# ECA
model_ECA = build_model(attn_type='eca', position='P3')
model_ECA, hist_ECA = compile_and_train(model_ECA, 'ECA')
res_ECA = evaluate_model(model_ECA, 'ECA@P3')
all_results['ECA'] = {**res_ECA, 'history': hist_ECA}

# CBAM
model_CBAM = build_model(attn_type='cbam', position='P3')
model_CBAM, hist_CBAM = compile_and_train(model_CBAM, 'CBAM')
res_CBAM = evaluate_model(model_CBAM, 'CBAM@P3')
all_results['CBAM'] = {**res_CBAM, 'history': hist_CBAM}

# %% [markdown]
# ## Cell 22 — Plot: Attention Mechanism vs Accuracy & Macro F1 (§24.3, §24.4)

# %%
mech_names   = ['Baseline', 'Channel/SE', 'Spatial', 'Chan+Sp', 'Self-Attn', 'ECA', 'CBAM']
mech_results = [res_A0, res_A1, res_A2, res_A3, res_A4, res_ECA, res_CBAM]
colors_bar   = plt.cm.Set2(np.linspace(0, 1, len(mech_names)))

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, metric, ylabel, title in zip(axes,
    ['acc', 'f1'],
    ['Test Accuracy', 'Macro F1'],
    ['Attention Mechanism vs Accuracy (§24.3)',
     'Attention Mechanism vs Macro F1 (§24.4)']):
    vals = [r[metric] for r in mech_results]
    bars = ax.bar(mech_names, vals, color=colors_bar, label='Attention Models')
    ax.axhline(res_A0['acc'], color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_xlabel('Attention Mechanism'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticklabels(mech_names, rotation=20, ha='right')
    ax.set_ylim(min(vals) - 0.05, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                f'{v:.3f}', ha='center', fontweight='bold')
    ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("plot_acc_f1_bar.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 23 — Plot: Accuracy vs Parameters (§24.5) & F1 vs Inference Time (§24.6)

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax = axes[0]
for nm, r, c in zip(mech_names, mech_results, colors_bar):
    ax.scatter(r['params']/1e6, r['acc'], s=180, color=c, zorder=5, label=nm)
    ax.annotate(nm, (r['params']/1e6, r['acc']),
                textcoords='offset points', xytext=(6, 4), fontsize=11)
ax.set_xlabel('Total Parameters (M)'); ax.set_ylabel('Test Accuracy')
ax.set_title('Accuracy–Complexity Trade-off (§24.5)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=10, loc='lower right')

ax = axes[1]
for nm, r, c in zip(mech_names, mech_results, colors_bar):
    ax.scatter(r['inf_time'], r['f1'], s=180, color=c, zorder=5, label=nm)
    ax.annotate(nm, (r['inf_time'], r['f1']),
                textcoords='offset points', xytext=(6, 4), fontsize=11)
ax.set_xlabel('Inference Time (ms/image)'); ax.set_ylabel('Macro F1-score')
ax.set_title('F1–Inference-Time Trade-off (§24.6)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=10, loc='lower left')

plt.tight_layout()
plt.savefig("plot_tradeoffs.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 24 — Confusion Matrices: Baseline vs Best Model (§25)

# %%
best_key = max(all_results, key=lambda k: all_results[k]['acc'])
best_r   = all_results[best_key]

def plot_cm(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, annot_kws={'size': 13})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot_cm(res_A0['y_true'], res_A0['y_pred'],
        'Baseline ResNet50', axes[0])
plot_cm(best_r['y_true'], best_r['y_pred'],
        f'Best Model ({best_r["name"]})', axes[1])
plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 25 — Internal Attention: Channel Weights Bar (§28.1)

# %%
# Grab one test image for all visualizations
sample_img_raw, sample_lbl_raw = next(iter(test_raw))
raw_disp = tf.image.resize(sample_img_raw, [224, 224]).numpy().astype(np.uint8)
proc_img = preprocess_fn(tf.cast(raw_disp, tf.float32))
proc_img = tf.expand_dims(proc_img, 0)

# Extract P3 features from A1 model
base_A1   = model_A1.layers[0]
feat_p3   = tf.keras.Model(inputs=base_A1.input,
                            outputs=base_A1.get_layer('conv4_block6_out').output)(proc_img)
ca_layer  = [l for l in model_A1.layers if isinstance(l, ChannelAttention)][0]
_, cw     = ca_layer(feat_p3)            # cw shape: (1,1,1,1024)
cw_np     = cw.numpy().squeeze()         # (1024,)

top_k    = 20
top_idx  = np.argsort(cw_np)[-top_k:][::-1]

fig, ax = plt.subplots(figsize=(16, 5))
bars = ax.bar(range(top_k), cw_np[top_idx], color='steelblue', label='Channel Weight')
ax.set_xticks(range(top_k))
ax.set_xticklabels([f'Ch {i}' for i in top_idx], rotation=45, ha='right', fontsize=11)
ax.set_xlabel('Channel Index'); ax.set_ylabel('Attention Weight')
ax.set_title(f'Top-{top_k} Channel Attention Weights (A1 @ P3)')
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("channel_attn_weights.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 26 — Internal Attention: Spatial Heatmap & Overlay (§28.2)

# %%
base_A2  = model_A2.layers[0]
feat_sp  = tf.keras.Model(inputs=base_A2.input,
                           outputs=base_A2.get_layer('conv4_block6_out').output)(proc_img)
sa_layer = [l for l in model_A2.layers if isinstance(l, SpatialAttention)][0]
_, smap  = sa_layer(feat_sp)                                      # (1,14,14,1)
smap_np  = smap.numpy().squeeze()                                  # (14,14)
smap_up  = tf.image.resize(smap[0], [224, 224]).numpy().squeeze()  # (224,224)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(raw_disp); axes[0].set_title('Input Image'); axes[0].axis('off')
im = axes[1].imshow(smap_np, cmap='jet')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].set_title('Spatial Attention Map (14×14)'); axes[1].axis('off')
axes[2].imshow(raw_disp)
axes[2].imshow(smap_up, cmap='jet', alpha=0.5)
axes[2].set_title('Spatial Attention Overlay'); axes[2].axis('off')
plt.suptitle('Spatial Attention Visualization — A2 @ P3', fontweight='bold')
plt.tight_layout()
plt.savefig("spatial_attn_map.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 27 — Internal Attention: Self-Attention Response (§28.3)

# %%
base_A4  = model_A4.layers[0]
feat_sa  = tf.keras.Model(inputs=base_A4.input,
                           outputs=base_A4.get_layer('conv4_block6_out').output)(proc_img)
sa_self  = [l for l in model_A4.layers if isinstance(l, SelfAttention)][0]
_, A_mat = sa_self(feat_sa)               # A_mat: (1, N, N)  N=14*14=196
mean_attn = A_mat.numpy().squeeze().mean(axis=0).reshape(14, 14)
mean_up   = tf.image.resize(mean_attn[..., None], [224, 224]).numpy().squeeze()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(raw_disp); axes[0].set_title('Input'); axes[0].axis('off')
im = axes[1].imshow(mean_attn, cmap='viridis')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].set_title('Self-Attn Mean Response (14×14)'); axes[1].axis('off')
axes[2].imshow(raw_disp)
axes[2].imshow(mean_up, cmap='viridis', alpha=0.5)
axes[2].set_title('Self-Attn Overlay'); axes[2].axis('off')
plt.suptitle('Self-Attention Response — A4 @ P3', fontweight='bold')
plt.tight_layout()
plt.savefig("self_attn_response.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 28 — Attention Maps at P2, P3, P4 (§29)

# %%
# Train spatial-attention-only models at P2 and P4 for visualization
model_sp_P2 = build_model(attn_type='spatial', position='P2')
model_sp_P2, _ = compile_and_train(model_sp_P2, 'sp_P2_vis')

model_sp_P4 = build_model(attn_type='spatial', position='P4')
model_sp_P4, _ = compile_and_train(model_sp_P4, 'sp_P4_vis')

def get_sp_map(model, proc_img, stage_name):
    base = model.layers[0]
    feat = tf.keras.Model(inputs=base.input,
                           outputs=base.get_layer(stage_name).output)(proc_img)
    sp_l = [l for l in model.layers if isinstance(l, SpatialAttention)][0]
    _, sm = sp_l(feat)
    sm_raw = sm.numpy().squeeze()
    sm_up  = tf.image.resize(sm[0], [224, 224]).numpy().squeeze()
    return sm_raw, sm_up

sp_P2_raw, sp_P2_up = get_sp_map(model_sp_P2, proc_img, 'conv2_block3_out')
sp_P3_raw, sp_P3_up = get_sp_map(model_A2,    proc_img, 'conv4_block6_out')
sp_P4_raw, sp_P4_up = get_sp_map(model_sp_P4, proc_img, 'conv5_block3_out')

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
row0 = [('Input', raw_disp, None),
        ('P2 Map\n(56×56)', sp_P2_raw, None),
        ('P3 Map\n(14×14)', sp_P3_raw, None),
        ('P4 Map\n(7×7)',   sp_P4_raw, None)]
row1 = [('Input', raw_disp, None),
        ('P2 Overlay', raw_disp, sp_P2_up),
        ('P3 Overlay', raw_disp, sp_P3_up),
        ('P4 Overlay', raw_disp, sp_P4_up)]

for col, (title, img, overlay) in enumerate(row0):
    axes[0, col].imshow(img, cmap='jet' if overlay is None and col > 0 else None)
    axes[0, col].set_title(title); axes[0, col].axis('off')

for col, (title, img, overlay) in enumerate(row1):
    axes[1, col].imshow(img)
    if overlay is not None:
        axes[1, col].imshow(overlay, cmap='jet', alpha=0.5)
    axes[1, col].set_title(title); axes[1, col].axis('off')

plt.suptitle('Spatial Attention Maps at P2, P3, P4 (§29)', fontweight='bold')
plt.tight_layout()
plt.savefig("attn_position_maps.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 29 — Feature Maps Before & After Attention (§31.5, §31.10)

# %%
base_A3   = model_A3.layers[0]
feat_p3_A3 = tf.keras.Model(inputs=base_A3.input,
                              outputs=base_A3.get_layer('conv4_block6_out').output)(proc_img)
feat_before = feat_p3_A3.numpy().squeeze()   # (14,14,1024)

ca_A3 = [l for l in model_A3.layers if isinstance(l, ChannelAttention)][0]
sa_A3 = [l for l in model_A3.layers if isinstance(l, SpatialAttention)][0]
f_ca, _ = ca_A3(feat_p3_A3)
feat_after_np = sa_A3(f_ca)[0].numpy().squeeze()   # (14,14,1024)

n_show = 8
step   = feat_before.shape[2] // n_show

fig, axes = plt.subplots(2, n_show, figsize=(22, 6))
for i in range(n_show):
    c = i * step
    axes[0, i].imshow(feat_before[:, :, c], cmap='viridis'); axes[0, i].axis('off')
    axes[1, i].imshow(feat_after_np[:, :, c],  cmap='viridis'); axes[1, i].axis('off')
axes[0, 0].set_title('Before Attention', fontsize=13, loc='left')
axes[1, 0].set_title('After Attention',  fontsize=13, loc='left')
plt.suptitle('Feature Maps Before & After Channel+Spatial Attention @ P3', fontweight='bold')
plt.tight_layout()
plt.savefig("feature_maps_before_after.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 30 — Grad-CAM Helper & Comparison (§27)

# %%
def grad_cam_from_model(model, proc_img):
    """Compute Grad-CAM using the last conv layer inside the ResNet50 sub-model."""
    base       = model.layers[0]
    target_lyr = base.get_layer('conv5_block3_out')
    cam_model  = tf.keras.Model(inputs=base.input,
                                 outputs=[target_lyr.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(proc_img)
        cls_idx = tf.argmax(preds[0])
        loss    = preds[:, cls_idx]
    grads  = tape.gradient(loss, conv_out)           # (1,7,7,2048)
    alpha  = tf.reduce_mean(grads, axis=[1, 2])      # (1,2048)
    cam    = tf.reduce_sum(conv_out * alpha[:, None, None, :], axis=-1)
    cam    = tf.nn.relu(cam).numpy().squeeze()
    cam    = cam / (cam.max() + 1e-8)
    cam_up = tf.image.resize(cam[..., None], [224, 224]).numpy().squeeze()
    return cam_up, int(cls_idx)

cam_bl,   pred_bl   = grad_cam_from_model(model_A0,   proc_img)
cam_A3,   pred_A3   = grad_cam_from_model(model_A3,   proc_img)
cam_CBAM, pred_CBAM = grad_cam_from_model(model_CBAM, proc_img)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
panels = [
    ('Input', raw_disp, None),
    (f'Baseline GradCAM\n(pred:{CLASS_NAMES[pred_bl]})',   raw_disp, cam_bl),
    (f'Chan+Sp GradCAM\n(pred:{CLASS_NAMES[pred_A3]})',    raw_disp, cam_A3),
    (f'CBAM GradCAM\n(pred:{CLASS_NAMES[pred_CBAM]})',     raw_disp, cam_CBAM),
]
for ax, (title, img, cam) in zip(axes, panels):
    ax.imshow(img)
    if cam is not None:
        ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.set_title(title); ax.axis('off')
plt.suptitle('Grad-CAM Comparison — Baseline vs Attention Models (§27)', fontweight='bold')
plt.tight_layout()
plt.savefig("gradcam_comparison.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 31 — Qualitative Comparison: 3 Correct + 2 Incorrect (§30)

# %%
def collect_examples(model, n_correct=3, n_wrong=2):
    correct_ex, wrong_ex = [], []
    for imgs, lbls in test_ds:
        preds_batch = np.argmax(model.predict(imgs, verbose=0), axis=1)
        for i in range(len(lbls)):
            true_l = int(lbls[i]); pred_l = int(preds_batch[i])
            disp = imgs[i].numpy() + [103.939, 116.779, 123.68]
            disp = np.clip(disp[..., ::-1], 0, 255).astype(np.uint8)
            if true_l == pred_l and len(correct_ex) < n_correct:
                correct_ex.append((disp, true_l, pred_l))
            elif true_l != pred_l and len(wrong_ex) < n_wrong:
                wrong_ex.append((disp, true_l, pred_l))
        if len(correct_ex) >= n_correct and len(wrong_ex) >= n_wrong:
            break
    return correct_ex + wrong_ex

examples = collect_examples(model_A3)
col_titles = ['Input', 'Baseline\nGrad-CAM', 'Channel\nGrad-CAM',
              'Spatial\nAttn Map', 'Chan+Spatial\nGrad-CAM']

fig, axes = plt.subplots(len(examples), 5, figsize=(24, len(examples) * 4 + 1))
axes[0, 0].set_title('Input', fontsize=14)
for c, t in zip(range(1, 5), col_titles[1:]):
    axes[0, c].set_title(t, fontsize=13)

for row, (img_disp, true_l, pred_l) in enumerate(examples):
    tag = 'Correct' if true_l == pred_l else 'Wrong'
    pi = tf.expand_dims(preprocess_fn(tf.cast(img_disp, tf.float32)), 0)

    axes[row, 0].imshow(img_disp)
    axes[row, 0].set_ylabel(f'{tag}\nTrue:{CLASS_NAMES[true_l]}\nPred:{CLASS_NAMES[pred_l]}',
                             fontsize=10)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    cam_bl_i, _ = grad_cam_from_model(model_A0, pi)
    axes[row, 1].imshow(img_disp); axes[row, 1].imshow(cam_bl_i, cmap='jet', alpha=0.5)
    axes[row, 1].axis('off')

    cam_ch_i, _ = grad_cam_from_model(model_A1, pi)
    axes[row, 2].imshow(img_disp); axes[row, 2].imshow(cam_ch_i, cmap='jet', alpha=0.5)
    axes[row, 2].axis('off')

    base_A2_i = model_A2.layers[0]
    fsp = tf.keras.Model(inputs=base_A2_i.input,
                          outputs=base_A2_i.get_layer('conv4_block6_out').output)(pi)
    sp_l_i = [l for l in model_A2.layers if isinstance(l, SpatialAttention)][0]
    _, sm_i = sp_l_i(fsp)
    sm_up_i = tf.image.resize(sm_i[0], [224, 224]).numpy().squeeze()
    axes[row, 3].imshow(img_disp); axes[row, 3].imshow(sm_up_i, cmap='jet', alpha=0.5)
    axes[row, 3].axis('off')

    cam_cs_i, _ = grad_cam_from_model(model_A3, pi)
    axes[row, 4].imshow(img_disp); axes[row, 4].imshow(cam_cs_i, cmap='jet', alpha=0.5)
    axes[row, 4].axis('off')

plt.suptitle('Qualitative Comparison — 3 Correct + 2 Incorrect Images (§30)',
             fontweight='bold')
plt.tight_layout()
plt.savefig("qualitative_comparison.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 32 — Additional Mechanism Comparison: 4-Panel (§19, §36)

# %%
add_names    = ['No Attention', 'Channel/SE', 'Spatial', 'ECA', 'CBAM', 'Self-Attn', 'Combined']
add_res_list = [res_A0, res_A1, res_A2, res_ECA, res_CBAM, res_A4, res_A5]
c_pal        = plt.cm.Paired(np.linspace(0, 1, len(add_names)))

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
for ax, metric, ylabel, title in zip(
    axes.flatten(),
    ['acc',    'f1',       'params',            'inf_time'],
    ['Accuracy','Macro F1','Parameters (M)',    'Inference Time (ms)'],
    ['Classification Accuracy', 'Macro F1-Score',
     'Model Parameters',        'Average Inference Time']):
    vals = [r[metric] / (1e6 if metric == 'params' else 1) for r in add_res_list]
    bars = ax.bar(add_names, vals, color=c_pal, label='Mechanism')
    ax.set_title(title); ax.set_xlabel('Attention Mechanism'); ax.set_ylabel(ylabel)
    ax.set_xticklabels(add_names, rotation=20, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right' if metric == 'params' else 'lower right')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.002,
                f'{v:.3f}' if metric != 'params' else f'{v:.1f}M',
                ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Additional Attention Mechanism Comparison @ P3 (§19, §36)',
             fontweight='bold')
plt.tight_layout()
plt.savefig("additional_attn_comparison.png", bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 33 — Summary Results Table (CSV)

# %%
rows = []
for key, label in [
    ('A0','Baseline'),    ('A1','Channel@P3'),  ('A2','Spatial@P3'),
    ('A3','Chan+Sp@P3'),  ('B1','Chan+Sp@P2'),  ('B3','Chan+Sp@P4'),
    ('C1','Chan+Sp@P3+P4'),('C2','Chan+Sp@P2+P3+P4'),
    ('A4','Self-Attn@P3'),('A5','Combined@P3'),
    ('ECA','ECA@P3'),     ('CBAM','CBAM@P3'),
]:
    if key in all_results:
        r = all_results[key]
        rows.append({'Model': label,
                     'Accuracy':  f"{r['acc']:.4f}",
                     'Precision': f"{r['prec']:.4f}",
                     'Recall':    f"{r['rec']:.4f}",
                     'Macro F1':  f"{r['f1']:.4f}",
                     'Params(M)': f"{r['params']/1e6:.2f}",
                     'Inf(ms)':   f"{r['inf_time']:.2f}"})

df = pd.DataFrame(rows)
print("\n===== MAIN RESULTS TABLE =====")
print(df.to_string(index=False))
df.to_csv("results_table.csv", index=False)
print("\nSaved: results_table.csv")

# %% [markdown]
# ## Cell 34 — Kernel Size Comparison Table & Final Summary

# %%
print("\n===== KERNEL SIZE ABLATION =====")
ker_rows = [{'Kernel': f'{k}×{k}',
             'Accuracy': f"{kernel_results[k]['acc']:.4f}",
             'Macro F1': f"{kernel_results[k]['f1']:.4f}",
             'Inf(ms)':  f"{kernel_results[k]['inf_time']:.2f}"}
            for k in [3, 5, 7]]
print(pd.DataFrame(ker_rows).to_string(index=False))

import glob
print("\n\nAll saved plots:")
for f in sorted(glob.glob("*.png")) + sorted(glob.glob("*.csv")):
    print(f"  • {f}")
print("\nExperiments complete!")