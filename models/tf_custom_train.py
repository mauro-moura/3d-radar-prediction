
import tensorflow as tf

@tf.function
def train_step(x, y_true, model, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)

        # Create nan mask
        nan_mask = tf.math.is_nan(y_true)
        nan_mask = tf.squeeze(nan_mask) # Remove extra dimensions if any

        y_pred_masked = tf.boolean_mask(y_pred, nan_mask)
        y_true_masked = tf.boolean_mask(y_true, nan_mask)

        loss = loss_fn(y_true_masked, y_pred_masked)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def custom_fit(model, x, y_true, num_epochs=100, loss_fn=tf.keras.losses.MeanSquaredError()):
    for i in range(num_epochs):
        loss = train_step(x, y_true, model, loss_fn)
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.numpy()}")
    return model