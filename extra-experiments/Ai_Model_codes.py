import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers












from tensorflow.keras import layers, models, regularizers
import tensorflow as tf





from tensorflow.keras import layers, models, regularizers
import tensorflow as tf





import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ============================================================
# Positional Encoding Layer
# ============================================================
class PositionalEncoding(layers.Layer):
    def __init__(self, time_steps: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.time_steps = time_steps
        self.d_model = d_model
        # Tell Keras this layer can pass through mask information
        self.supports_masking = True

    def build(self, input_shape):
        # Learnable positional embedding: (1, T, D)
        self.pos_emb = self.add_weight(
            name="pos_emb_weights",
            shape=(1, self.time_steps, self.d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x, mask=None):
        # x: (B, T, D), pos_emb: (1, T, D) -> broadcast add over batch
        return x + self.pos_emb

    def compute_mask(self, inputs, mask=None):
        # Preserve incoming mask (if any)
        return mask



def transformer_block(x, d_model=512, d_ff=1024, num_heads=8, dropout_rate=0.3, name_prefix="transf"):
    reg = regularizers.l2(1e-5)

    # LayerNorm + MHA + residual
    attn_input = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f"{name_prefix}_mha",
    )(attn_input, attn_input)
    attn_output = layers.Dropout(dropout_rate, name=f"{name_prefix}_attn_dropout")(attn_output)
    x = layers.Add(name=f"{name_prefix}_attn_residual")([x, attn_output])

    # LayerNorm + FFN + residual
    ffn_input = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x)
    ffn = layers.Dense(d_ff, activation="relu", kernel_regularizer=reg,
                       name=f"{name_prefix}_ffn_dense1")(ffn_input)
    ffn = layers.Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout")(ffn)
    ffn = layers.Dense(d_model, kernel_regularizer=reg,
                       name=f"{name_prefix}_ffn_dense2")(ffn)
    x = layers.Add(name=f"{name_prefix}_ffn_residual")([x, ffn])

    return x


def build_mwps_ultra_classifier(
    time_steps: int,
    n_features: int,
    num_classes: int,
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.3,
    num_transformer_blocks: int = 4,
) -> tf.keras.Model:
    """
    Ultra MWPS classifier: Dense projection + Transformer encoder stack + BiLSTM + attention pooling.
    All layers are standard keras.layers (no custom Layer subclasses, no Lambda).
    """
    reg = regularizers.l2(l2_reg)
    d_model = 512
    d_ff = 1024
    num_heads = 8

    inp = layers.Input(shape=(time_steps, n_features), name="X_seq")

    # Optional masking
    x = layers.Masking(mask_value=0.0, name="mask")(inp)

    # Dense projection to d_model
    x = layers.Dense(
        d_model,
        activation="linear",
        kernel_regularizer=reg,
        name="proj_dense",
    )(x)

    # Positional embeddings (trainable), added to x
    # positions: (T,)
    positions = tf.range(start=0, limit=time_steps, delta=1, dtype=tf.int32)
    pos_emb_layer = layers.Embedding(
        input_dim=time_steps,
        output_dim=d_model,
        name="pos_embedding",
    )
    # (T, d_model) -> (1, T, d_model) for broadcasting
    pos_emb = pos_emb_layer(positions)          # (T, d_model)
    pos_emb = tf.expand_dims(pos_emb, axis=0)   # (1, T, d_model)
    x = layers.Add(name="add_positional_encoding")([x, pos_emb])

    # Transformer encoder stack
    for i in range(num_transformer_blocks):
        x = transformer_block(
            x,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name_prefix=f"transf_block_{i+1}",
        )

    # BiLSTM over transformer outputs (return_sequences=True for attention)
    x = layers.Bidirectional(
        layers.LSTM(
            256,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)  # (B, T, 512)

    # --- Attention pooling over time, no Lambda ---
    # attn_scores: (B, T, 128) -> (B, T, 1)
    attn_scores = layers.Dense(
        128,
        activation="tanh",
        kernel_regularizer=reg,
        name="attn_dense",
    )(x)
    attn_scores = layers.Dense(
        1,
        kernel_regularizer=reg,
        name="attn_scores",
    )(attn_scores)  # (B, T, 1)

    attn_weights = layers.Softmax(axis=1, name="attn_softmax")(attn_scores)  # (B, T, 1)

    # Convert (B, T, 1) -> (B, 1, T) using Permute (built-in)
    attn_weights_T = layers.Permute((2, 1), name="attn_permute")(attn_weights)  # (B, 1, T)

    # Context = sum_t w_t * x_t using Dot (built-in)
    context = layers.Dot(axes=[2, 1], name="attn_dot")([attn_weights_T, x])  # (B, 1, 512)

    # Flatten context to (B, 512)
    x = layers.Reshape((d_model,), name="attn_context")(context)

    # Dense head
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="MWPS_Ultra_TransformerBiLSTM")
    return model




















def build_mwps_heavy_classifier(
    time_steps: int,
    n_features: int,
    num_classes: int,
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Strong LIGHT MWPS classifier (~5M params, 3D input).

    Input:  (batch, time_steps, n_features)
    Output: class probabilities (num_classes) via softmax

    Architecture:
      - Conv1D(256, k=7) + BN
      - Conv1D(256, k=5) + BN
      - BiLSTM(320) return_sequences=True
      - BiLSTM(320) return_sequences=False
      - Dense 512 → Dense 256 → Softmax(num_classes)
    """

    reg = regularizers.l2(l2_reg)

    inp = layers.Input(shape=(time_steps, n_features), name="X_seq")

    # Optional masking if you ever pad with zeros
    x = layers.Masking(mask_value=0.0, name="mask")(inp)

    # ---- Conv stack (temporal patterns) ----
    x = layers.Conv1D(
        filters=256,
        kernel_size=7,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)

    x = layers.Conv1D(
        filters=256,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)

    # ---- BiLSTM stack (big, ~5M total with dense head) ----
    x = layers.Bidirectional(
        layers.LSTM(
            320,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            320,
            return_sequences=False,
            kernel_regularizer=reg,
            name="lstm_2",
        ),
        name="bilstm_2",
    )(x)  # -> (batch, 640)

    # ---- Dense head ----
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="MWPS_LightClassifier_5M")
    return model











# ============================================================
# 2) LIGHT MWPS classifier (BiLSTM-based, 3D input)
# ============================================================
def build_mwps_light_classifier(
    time_steps: int,
    n_features: int,
    num_classes: int,
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Input:  (batch, time_steps, n_features)
    Output: class probabilities (num_classes) via softmax
    """

    reg = regularizers.l2(l2_reg)

    inp = layers.Input(shape=(time_steps, n_features), name="X_seq")

    # Optional masking if you ever pad with zeros
    x = layers.Masking(mask_value=0.0, name="mask")(inp)

    # 1D Conv block for local temporal patterns
    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)

    # BiLSTM stack
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=False,
            kernel_regularizer=reg,
            name="lstm_2",
        ),
        name="bilstm_2",
    )(x)

    # Dense head
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="MWPS_LightClassifier")
    return model


# ============================================================
# 3) Distance-aware custom loss
# ============================================================
def make_distance_aware_loss(label_values, alpha: float = 1.0):
    """
    label_values: list/np.array of numeric label values (e.g. [-10, -9, ..., 10])
                  indexed by class index (0..num_classes-1)

    alpha: weight for distance penalty term.
           Larger alpha => stronger punishment for far misclassifications.
    """

    label_values = tf.constant(label_values, dtype=tf.float32)  # (C,)

    def loss_fn(y_true, y_pred):
        """
        y_true: integer class indices (SparseCategorical style),
                shape (batch,) or (batch, 1)
        y_pred: softmax probabilities, shape (batch, C)
        """
        # Ensure shapes: flatten to (batch,)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Standard sparse categorical cross-entropy
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False,  # because last layer is softmax
        )  # (batch,)

        # Numeric value of true label: label_values[y_true]
        true_vals = tf.gather(label_values, y_true)  # (batch,)

        # Expected numeric value under predicted distribution:
        # pred_vals[i] = sum_c p[i,c] * label_values[c]
        pred_vals = tf.tensordot(y_pred, label_values, axes=1)  # (batch,)

        # Distance penalty: |true_numeric - pred_numeric|
        dist = tf.abs(true_vals - pred_vals)  # (batch,)

        return ce + alpha * dist

    return loss_fn







def build_mwps_strong_classifier(
    time_steps: int,
    n_features: int,
    num_classes: int,
    l2_reg: float = 1e-5,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Strong MWPS classifier (~1.6M params).

    Input:  (batch, time_steps, n_features)
    Output: class probabilities (num_classes) via softmax
    """
    reg = regularizers.l2(l2_reg)

    inp = layers.Input(shape=(time_steps, n_features), name="X_seq")

    # ---- Conv stack ----
    x = layers.Conv1D(
        filters=256,
        kernel_size=7,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(inp)
    x = layers.BatchNormalization(name="bn_conv1")(x)

    x = layers.Conv1D(
        filters=256,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=reg,
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_conv")(x)

    # ---- BiLSTM stack ----
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=reg,
            name="lstm_2",
        ),
        name="bilstm_2",
    )(x)

    # ---- Attention over time (no Lambda) ----
    attn_scores = layers.Dense(128, activation="tanh", name="attn_dense")(x)
    attn_scores = layers.Dense(1, name="attn_scores")(attn_scores)  # (B, T, 1)
    attn_weights = layers.Softmax(axis=1, name="attn_softmax")(attn_scores)
    x = layers.Multiply(name="attn_weighted")([x, attn_weights])

    # Single pooling step
    x = layers.GlobalAveragePooling1D(name="attn_pool")(x)  # (B, 256)

    # ---- Dense head ----
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = models.Model(inputs=inp, outputs=out, name="MWPS_StrongClassifier")
    return model








import numpy as np
import tensorflow as tf



# ============================================================
# CONFIG
# ============================================================
TIME_STEPS = 200
N_FEATURES = 132
NUM_CLASSES = 21                     # ZZL3..ZZL-3 or similar
LABEL_VALUES = np.arange(-10, 11)    # Example: -10 to +10
ALPHA = 1.0
LEARNING_RATE = 1e-4


# ============================================================
# Helper: Build + Compile Model
# ============================================================
def build_and_compile(model_fn, name: str):
    print("\n==============================")
    print(f"Building model: {name}")
    print("==============================")

    model = model_fn(
        time_steps=TIME_STEPS,
        n_features=N_FEATURES,
        num_classes=NUM_CLASSES,
    )

    loss_fn = make_distance_aware_loss(LABEL_VALUES, alpha=ALPHA)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    model.summary()

    return model


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # ---- LIGHT Model ----
    light_model = build_and_compile(
        build_mwps_light_classifier,
        name="MWPS_LightClassifier"
    )

    # ---- STRONG Model ----
    strong_model = build_and_compile(
        build_mwps_strong_classifier,
        name="MWPS_StrongClassifier"
    )

    # ---- HEAVY Model (~5M params) ----
    heavy_model = build_and_compile(
        build_mwps_heavy_classifier,
        name="MWPS_HeavyClassifier_5M"
    )

    # ---- ULTRA Model (Transformer + BiLSTM, 10–15M params) ----
    ultra_model = build_and_compile(
        build_mwps_ultra_classifier,
        name="MWPS_Ultra_TransformerBiLSTM"
    )

    print("\nAll models built successfully!\n")


    # ============================================================
    # OPTIONAL: TRAIN MODELS (UNCOMMENT TO TRAIN)
    # ============================================================
    """
    X_train = np.load("X_train.npy")    # shape (B, TIME_STEPS, N_FEATURES)
    y_train = np.load("y_train.npy")    # shape (B,)

    ultra_model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        verbose=1,
    )
    """



