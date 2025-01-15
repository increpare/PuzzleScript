import os
import hydra
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import json
import glob
from PIL import Image
import numpy as np
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


@dataclass
class Config:
    # Collect new transitions
    collect_transitions: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


# Model definition
class WorldModel(nn.Module):
    @nn.compact
    def __call__(self, image, action):
        # Encoder
        x = nn.Conv(features=32, kernel_size=(3, 3))(image)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        
        # Combine with action
        action_embedded = nn.Dense(features=64)(action)
        x = jnp.concatenate([x, action_embedded], axis=1)
        
        # Decoder
        x = nn.Dense(features=64 * 64)(x)
        x = x.reshape((-1, 8, 8, 64))
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3))(x)
        return x

# Data loading
def load_data(base_path):
    with open(f"{base_path}/transitions.json") as f:
        transitions_txt = f.read()
        transitions_txt = f"[{transitions_txt.replace('}\n{', '},\n{')}]"
        try:
            transitions = json.loads(transitions_txt)
        except json.JSONDecodeError as e:
            print(f"Error loading {base_path}: {e}")
            return None, None
    
    image_pairs = []
    actions = []
    
    for td in transitions:
        img_a, img_b, action = td["state1"], td["state2"], td["action"]
        img_a_path = f"{base_path}/images/{img_a}.png"
        img_b_path = f"{base_path}/images/{img_b}.png"
        
        img_a = np.array(Image.open(img_a_path).resize((64, 64))) / 255.0
        img_b = np.array(Image.open(img_b_path).resize((64, 64))) / 255.0
        
        image_pairs.append((img_a, img_b))
        actions.append(action)
    
    return np.array(image_pairs), np.array(actions)

# Training setup
def create_train_state(rng, model):
    params = model.init(rng, jnp.ones((1, 64, 64, 3)), jnp.ones((1, 1)))
    tx = optax.adam(learning_rate=1e-3)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_images, batch_actions, batch_targets):
    def loss_fn(params):
        predictions = state.apply_fn(params, batch_images, batch_actions)
        loss = jnp.mean((predictions - batch_targets) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

TRANS_DIR = "transitions"

def collect_transitions(trans_path):
    images_a = []
    images_b = []
    actions = []

    game_hashes = os.listdir(TRANS_DIR)
    for game_hash in game_hashes:
        print(f"Collecting transitions for {game_hash}")
        game_dir = os.path.join("transitions", game_hash)
        level_ns = os.listdir(game_dir)
        for level_n in level_ns:
            # Load data
            image_pairs, lvl_actions = load_data(f"transitions/{game_hash}/{level_n}")
            if image_pairs is None:
                continue
            lvl_images_a = image_pairs[:, 0]
            lvl_images_b = image_pairs[:, 1]
            lvl_actions = lvl_actions.reshape(-1, 1)

            images_a.append(lvl_images_a)
            images_b.append(lvl_images_b)
            actions.append(lvl_actions)

    images_a = np.concatenate(images_a)
    images_b = np.concatenate(images_b)

    # Save with numpy
    np.savez(trans_path, images_a=images_a, images_b=images_b, actions=actions)

# Main training loop
@hydra.main(config_name="config", version_base="1.3")
def train(cfg: Config):
    rng = jax.random.PRNGKey(0)
    model = WorldModel()

    trans_path = "transitions.npz"

    if cfg.collect_transitions or not os.path.isfile(trans_path):
        collect_transitions(trans_path)
    else:
        data = np.load(trans_path)
        images_a = data["images_a"]
        images_b = data["images_b"]
        actions = data["actions"]

    print(f"Loaded {len(images_a)} transitions")
    
    # Create training state
    state = create_train_state(rng, model)
    
    # Training loop
    batch_size = 32
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Shuffle data
        perm = jax.random.permutation(rng, len(images_a))
        images_a = images_a[perm]
        images_b = images_b[perm]
        actions = actions[perm]
        
        # Batch training
        for i in range(0, len(images_a), batch_size):
            batch_images = images_a[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_targets = images_b[i:i+batch_size]
            
            state, loss = train_step(state, batch_images, batch_actions, batch_targets)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return state

if __name__ == "__main__":
    train()