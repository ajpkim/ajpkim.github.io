---
layout: default
title: "AlphaConnect: an AlphaZero inspired deep RL system"
---

# Implementing an AlphaZero inspired deep RL system <!-- omit in toc -->

September, 2020  
[github]{:target="_blank"}


[github]: https://github.com/ajpkim/alpha_connect4

---
## 1. AlphaGo &rarr; AlphaGo Zero &rarr; **AlphaZero** &rarr; MuZero 

In 2015 DeepMind's [AlphaGo]{:target="_blank"} beat one of the most decorated [Go]{:target="_blank"} players of all time, Lee Sedol, 4-1 in a highly publicized match. The event brought Go within the domain of computer dominated games far earlier than many predicted given the complexity of the game. While the rules of Go are extremely simple, the number of possible board states that follow is over $10^{170}$, an enormous number that illuminates why no computer program had ever beaten a professional Go player prior to AlphaGo. The vast complexity of the game and the futility of traditional search based methods leads many to perceive Go as the pinnacle of human intuition and strategy within the domain of games. 

[AlphaGo]: https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
[Go]: https://en.wikipedia.org/wiki/Go_(game)

DeepMind overcame the traditional obstacles posed by Go by pairing a deep neural network with a search algorithm that leverages the network's intuitive evaluations of the game state to focus the search on fruitful parts of the immense game tree. This setup, combined with a novel approach to improving the network with training data generated through self-play, allowed AlphaGo to surpass human level performance in the game of Go. Subsequent versions of AlphaGo achieved even greater performance using simpler and more generalizable techniques. The major achievements at each step were the removal of reliance on human game data to bootstrap training ([AlphaGo Zero][AlphaGo Zero link]{:target="_blank"}), the removal of game specific heuristics in training and generalization to Chess and Shogi ([AlphaZero]{:target="_blank"}), and the replacement of a given deterministic model by a learned internal model - rules and all - along with further domain generalization ([MuZero]{:target="_blank"}).

[AlphaGo Zero link]: https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ
[AlphaZero]: https://arxiv.org/abs/1712.01815
[MuZero]: https://arxiv.org/abs/1911.08265

These innovations all support a learning system that is capable of bootstrapping knowledge from its own experience and achieving mastery across many environments without needing any hard-coded domain knowledge or initial training data. To better understand how these systems work I recently implemented my own AlphaZero inspired deep RL system that learns [connect4]{:target="_blank"} entirely through self-play and which can be extended to other adversarial games. Here, I'll walk through the key concepts (move selection and MCTS, self-play, training, model architecture) behind AlphaZero with code examples using pytorch. My implementation is a shallower version of the original architecture with some minor differences which I will point out as they come up. 

[connect4]: https://en.wikipedia.org/wiki/Connect_Four


---
<div id="side-toc">

# Table of Contents <!-- omit in toc -->
- [1. AlphaGo &rarr; AlphaGo Zero &rarr; **AlphaZero** &rarr; MuZero](#1-alphago--alphago-zero--alphazero--muzero)
- [2. The Intuitive Picture](#2-the-intuitive-picture)
- [3. Move Selection](#3-move-selection)
  - [3.1. MCTS simulation](#31-mcts-simulation)
- [4. Self-Play Policy Improvement](#4-self-play-policy-improvement)
  - [4.1. MCTS self-play](#41-mcts-self-play)
  - [4.2. Loss and learning](#42-loss-and-learning)
- [5. The Network](#5-the-network)
  - [5.1. Game state encoding](#51-game-state-encoding)
  - [5.2. Convolutional layer](#52-convolutional-layer)
  - [5.3. Residual blocks](#53-residual-blocks)
  - [5.4. Value head](#54-value-head)
  - [5.5. Policy head](#55-policy-head)
  - [5.6. Putting it all together](#56-putting-it-all-together)
- [6. AlphaConnect](#6-alphaconnect)
  - [6.1. Hyperparameters](#61-hyperparameters)
  - [6.2. Multithreading](#62-multithreading)
- [7. Looking Ahead](#7-looking-ahead)

</div>


# Table of Contents <!-- omit in toc -->

- [1. AlphaGo &rarr; AlphaGo Zero &rarr; **AlphaZero** &rarr; MuZero](#1-alphago--alphago-zero--alphazero--muzero)
- [2. The Intuitive Picture](#2-the-intuitive-picture)
- [3. Move Selection](#3-move-selection)
  - [3.1. MCTS simulation](#31-mcts-simulation)
- [4. Self-Play Policy Improvement](#4-self-play-policy-improvement)
  - [4.1. MCTS self-play](#41-mcts-self-play)
  - [4.2. Loss and learning](#42-loss-and-learning)
- [5. The Network](#5-the-network)
  - [5.1. Game state encoding](#51-game-state-encoding)
  - [5.2. Convolutional layer](#52-convolutional-layer)
  - [5.3. Residual blocks](#53-residual-blocks)
  - [5.4. Value head](#54-value-head)
  - [5.5. Policy head](#55-policy-head)
  - [5.6. Putting it all together](#56-putting-it-all-together)
- [6. AlphaConnect](#6-alphaconnect)
  - [6.1. Hyperparameters](#61-hyperparameters)
  - [6.2. Multithreading](#62-multithreading)
- [7. Looking Ahead](#7-looking-ahead)

---

## 2. The Intuitive Picture


Before discussing the details I want to give a little more overview of how AlphaZero works at an intuitive level to help frame what follows. AlphaZero learns and plays the game in a manner more familiar to humans than previous game engines like [Deep Blue][Deep Blue]{:target="_blank"} or [StockFish]{:target="_blank"} which rely on much deeper search and hard-coded game-specific evaluation features. In contrast, AlphaZero learns entirely from its own experience playing the game, adopting strategies that fare well and moving on from ones that fare poorly.

[Deep Blue]: https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)
[Stockfish]: https://en.wikipedia.org/wiki/Stockfish_(chess)

The system begins with no strategic information about the game and must discover meaningful patterns and strategies by itself. Over the course of training we can watch AlphaZero discover openings and tactics that have been used at different times throughout human history as the system explores the space of possibilities. The learning and creativity in devising new strategies in games as complex as chess and Go is remarkable.

When making a move, AlphaZero begins by querying its deep neural network for an approximate evaluation of the game state and a preliminary probability distribution over the possible actions. The initial move probabilities represent how attractive the network believes each move to be. These prior probabilities and value estimates are used to guide a more reflective search process. The search process explores the game tree, favoring paths that lead to higher state values and storing information about each state and possible action as it explores. The normalized action counts from the search process form an improved policy $\pi$ and are used to select the actual game move. Training aims to bring the network's intuitive value estimations of the game state closer to the actual game outcomes and the prior probabilities closer to the probabilities reflected in the normalized action counts from the search process. 

The result is a feedback cycle where the network guides the search process which in turn guides the network. This setup allows the system to generate its own training data through games against itself. The network is always challenged to improve by an equally skilled opponent and is unconstrained by human approaches and domain knowledge. Below I will discuss in more detail the move selection process, network architecture, and self-play training regime.

<br>

## 3. Move Selection

To select moves, AlphaZero uses a [Monte-Carlo tree search (MCTS)]{:target="_blank"} algorithm that is guided by the network's prior action probabilities $p$ and estimations of state value $v$:

[Monte-Carlo tree search (MCTS)]: https://www.youtube.com/watch?v=UXW2yZndl7U


$$(p,v)= f_{\theta} (s)$$

The search tree is used to acquire information about possible trajectories and is maintained throughout the course of the game. Nodes in the game tree correspond to game states and are connected by directed edges which represent actions. The following statistics are maintained for each edge and are used to decide which move to take during the MCTS search process:

  - $N(s,a)$: number of times action $a$ has been taken from state $s$
  - $W(s,a)$: total value derived after taking action $a$ in state $s$
  - $Q(s,a)$: mean value of taking action $a$ in state $s$
  - $P(s,a)$: prior probability of selecting action $a$ in state $s$ 


 When prompted for a move, AlphaZero executes 800 MCTS simulations. Each simulation begins with the current game state as the root and traverses the game tree according to a polynomial upper-confidence tree search algorithm (PUCT), a variant of the upper-confidence tree search algorithm (UCT), until a leaf node is encountered  (see below). A leaf node represents either a terminal game state or a game state not yet included in the tree. 
 
 Unlike traditional MCTS methods, AlphaZero does not use random rollouts to evaluate non-terminal leaf nodes. Instead, the network provides an approximate state value for the leaf node along with prior probabilities that the new leaf edges are initialized with. After initializing the new edges, the value of the leaf state (network approximation $v$ or actual game outcome) is propagated back up to each node along the path taken from the root to leaf. Other statistics are updated during this step as well. Winning paths accrue higher value (larger $W$ and $Q$) and are more likely to be explored in future simulations.
 
 Eventually, the normalized action counts (computed using $N$ values) from the MCTS simulations form an improved policy ${\pi}$. These action counts reflect the perceived value associated with each edge from the current root node. Winning paths should have high visit counts and thus have a greater probability of being selected when sampling from $\pi$. After executing 800 MCTS simulations and gathering statistics about the value of each move, the actual game move is then sampled from ${\pi}$. In competitive play, as opposed to training, the move with the highest visit count can be chosen deterministically. In my implementation I used 200 MCTS simulations given time and compute costs and the simplicity of connect4 in comparison to Go or chess. 



### 3.1. MCTS simulation

The goal of the MCTS simulations is to improve on the network's preliminary action probabilities and discover paths that lead to higher state values and winning the game. To accomplish this the PUCT algorithm must balance exploring possible moves with exploiting what it already believes are worthwhile paths. At each decision point the algorithm selects the move that maximizes $Q + U$ where $Q$ is the mean value of the next state and $U$ is function of $P$ and $N$ which controls the exploration vs. exploitation tradeoff. Initially $U$ dominates and forces exploration, but as more simulations are executed $Q$ comes to dominate and states with higher values are selected: 

<br>

$$
a = \argmax_a \bigg(Q(s,a) + U(s,a)\bigg)
$$


$$
U(s,a) = C_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}
$$

<br>

- $\sum_b N(s,b)$ is a sum over all the edges from the current state.
- $C_{puct}$ is a hyperparameter. A higher value encourages exploration. AlphaZero uses a value 1.0. 

<br>

In python (nodes hold stats N, W, Q about the edge leading from parent to self):

``` python
class Node:
    def __init__(self, state: np.array, player_id: int, parent=None):
        self.player_id = player_id # necessary to know player perspective
        self.state = state  # game state
        self.parent = parent
        self.edges = {}  # {k=action: v=child node}
        self.W = 0  # sum of total value derived below node
        self.N = 0  # visit count
        self.P = None  # prior probabilities for selecting edges (actions). {k=edge: v=prob}
    
    @property
    def Q(self):
        """Average value derived below node."""
        return self.W / self.N if self.N > 0 else 0

    @property
    def explored(self):
        """Boolean for whether node has been explored. False for all terminal nodes."""
        return True if self.edges else False

    @ property
    def Pi(self):
        """Improved MCTS-based policy derived from normalized action counts."""
        policy_probs = np.zeros(7) # adjust to fit game action space
        if self.N <= 1:  # no paths below node yet explored -> no policy. 
            return policy_probs
        # compute normalized action counts for each valid move from current node
        for action in range(len(policy_probs)):  
            if action in self.edges:
                policy_probs[action] = (self.edges[action].N / (self.N-1))
        return policy_probs


def select_leaf(node, game,  C_puct=1.0) -> Node:
    """Find a leaf node by recursively traversing the game tree"""
    # base case: return discovered leaf node.
    if not node.explored:
        return node
    
    # recursively take actions that maximize Q + U until a leaf node is found.
    highest_score = -float('inf')
    next_action = None
    for action in node.edges:
        Q = node.edges[action].Q
        U = C_puct * node.P[action] * (np.sqrt(node.N) / (1 + node.edges[action].N))
        if Q + U > highest_score:
            highest_score = Q + U
            next_action = action
    
    game.make_move(next_action)
    next_node = node.edges[next_action]

    return select_leaf(next_node, game)    

```

<br>


And here's a high level view of the whole MCTS process (```dirichlet_alpha``` is a hyperparameter impacting exploration by adding noise to the prior probabilities): 
```python
def mcts_search(root, net, game, n_simulations, C_puct, dirichlet_alpha, training)
    """
    Return game action after executing given number of MCTS simulations
    from root node (current game state).
    """
    for simulation in range(n_simulations):
        # Create game copy that can be manipulated for each simulation.
        game_copy = copy.deepcopy(game)  
        # Recursively execute PUCT algorithm described above in traversing tree to find leaf.
        leaf = select_leaf(root, game_copy, C_puct) 
        # Get state value, initialize edges, backfill and update statistics in path from root to leaf.
        process_leaf(leaf, net, game_copy, dirichlet_alpha)  
    # Sample action from MCTS-based policy or choose most visited edge deterministically.
    action = select_action(root, training) 

    return action

```

<br>

## 4. Self-Play Policy Improvement
The self-play training regime is what I find to be the most interesting part of AlphaZero. The ability to learn solely from its own experience playing games against itself  allows for more general learning that is unconstrained by human knowledge of the domain. AlphaZero requires no human feedback, hard-coded evaluation features, nor initial game data to achieve superhuman ability.  

### 4.1. MCTS self-play
The network is randomly initialized and entirely trained on data generated through games of self-play using the guided MCTS search process described above to select moves and find a policy $\pi$. For each time-step $t$ of a self-play game, a training example $(s_t, \pi_t, z_t)$ consisting of game state, MCTS policy, and game outcome from the perspective of the player at step $t$ is generated and stored. Game outcomes are necessarily added after the game has been completed.  A replay buffer stores all the training data from recent games to help reduce correlations in the training data and support more stable learning.

```python
def mcts_self_play(net, game, n_simulations, C_puct, dirichlet_alpha) -> tuple:
    """
    Generate training data via self-play. Returns tuple of (state, Pi, Z) lists.
    Pi: improved action probabilities resulting from MCTS.
    Z: game outcome with value in [-1, 0, 1] for loss, tie, win.
    """
    states, Pis, Zs = [],[],[]
    current_node = Node(game.state, parent=None, player_id=game.player_turn)

    while not game.outcome:

        action = mcts_search(current_node, net, game, n_simulations, C_puct, dirichlet_alpha, training=True)
        states.append(game.state)
        Pis.append(current_node.Pi)  # MCTS policy derived from actions taken in search
        Zs.append(0)  #  value for a tie game used as placeholder
        
        game.make_move(action)
        current_node = current_node.edges[action]

    # assign game outcome based on player perspective 
    if game.outcome == 1:  # player 1 won game
        Zs[::2] = [1] * len(Zs[::2])
        Zs[1::2] = [-1] * len(Zs[1::2])
    elif game.outcome == 2:  # player 2 won game
        Zs[::2] = [-1] * len(Zs[::2])
        Zs[1::2] = [1] * len(Zs[1::2])
    
    return states, Pis, Zs
```

<br>

### 4.2. Loss and learning
After some number of self-play games, a batch of training examples is drawn uniformly from the replay buffer and used to train the network. Recall that the network outputs a tuple of prior probabilities and a game outcome prediction given a state, $(p, v) = f_\theta (s)$. The relevant quantities for training are:


- $z$: Game outcome (+1: win, -1: loss, 0: tie)
- $v$: Network's predicted game outcome
- $\pi$: Improved MCTS-based policy
- $p$: Network's initial policy 
  
The network parameters $\theta$ are trained to minimize the error between the actual game outcome and the predicted outcome and to maximize the similarity between the improved MCTS-based policy and the network's intuitive policy. The loss function used to accomplish this is a combination of mean squared error for the value estimates and cross-entropy for the policy with a hyperparameter $c$ controlling L2 weight regularization:

$$
Loss = (z-v)^2 - {\pi} \log{p} + c \vert\vert\theta\vert\vert^2
$$


The improved network parameters $\theta$ are then used in generating the next iteration of self-play data which allow the network to better guide the MCTS which in turn leads to better training data.


In python (with regularization outsourced to the torch optimizer):

```python
class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, z, pi, v, p):
        value_loss = ((z - v) ** 2).mean()
        policy_loss = torch.sum(-pi * p.log(), axis=1).mean()
        return value_loss + policy_loss

def learn():
    """
    Perform one learning step with batch uniformly sampled from replay 
    buffer which stores data as named tuples.
    """
    batch = replay_buffer.sample(batch_size)
    states = torch.stack([x.state for x in batch]).to(device)
    pi = torch.tensor([x.pi for x in batch], dtype=torch.float32).to(device)
    z = torch.tensor([x.z for x in batch], dtype=torch.float32).to(device)

    optimizer.zero_grad() 
    v, p = net(states)
    loss = loss_fn(z, pi, v, p)
    loss.backward()
    optimizer.step()
    
```

<br>

## 5. The Network
With everything else now in place let's turn to the actual network architecture. AlphaZero is a deep residual neural network with value and policy heads. The network takes in the raw game state and outputs a tuple consisting of a preliminary policy vector of probabilities over the action space and a value estimate of the expected outcome (win, lose, draw). There are 4 main components to the AlphaZero architecture:
   1. convolutional layer
   2. residual blocks
   3. value head
   4. policy head

### 5.1. Game state encoding
Go is played on a 19x19 board and AlphaZero is provided the game state as a (19x19x17) tensor in the case of Go (different games require different dimensions). The 17 channels encode the position of each players pieces in the current and previous 7 positions as well as whose  turn it is. The rules of Go demand that historical board information is included in the input as the current board state alone isn't sufficient to describe the global game state. In my implementation I encoded the game state as a (6x7x3) tensor given that connect4 is played on a 6x7 grid and historical information doesn't impact the game state. Two separate channels represent each player's pieces (0 indicating empty, and 1 indicating a piece at the position), and the last channel is used to indicate whose turn it is.

### 5.2. Convolutional layer
The pairing of a deep convolutional neural network and reinforcement learning in [solving Atari games][solving Atari games]{:target="_blank"} using raw pixel data was an earlier breakthrough from DeepMind. AlphaZero similarly leverages convolutional layers throughout the network to help extract useful features from the raw game state. The initial component of the network is a sole convolutional block.

[solving Atari games]: https://arxiv.org/pdf/1312.5602v1.pdf

Here is the structure of AlphaZero's initial convolutional layer:
   - Raw game state &rarr; 256 convolutional filters (3x3) &rarr; batch norm &rarr; ReLU 

Implementation in pytorch (with less convolutional filters):


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=126, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=126)

    def forward(self, x):
        x = x.view(-1, 3, 6, 7)  # flexible batch sizes
        x = F.relu(self.bn1(self.conv1(x)))
        return x

```

<br>

### 5.3. Residual blocks
The majority of the network consists of residual blocks. These blocks employ skip connections which feed the outputs of each layer directly into the next layer as well as into layers that are deeper in the network. Skip connections protect against degradation that can occur as networks get deeper and are used throughout the AlphaZero architecture.

Here is the structure for each of AlphaZero's 40 residual blocks:
   - input &rarr; 256 convolutional filters (3x3) &rarr; batch norm &rarr; ReLU &rarr; 256 convolutional filters (3x3) &rarr; batch norm &rarr; skip connection &rarr; ReLU

Implementation in pytorch (again with less convolutional filters):

```python
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=126, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=126)
        self.conv2 = nn.Conv2d(in_channels=126, out_channels=126, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=126)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # skip connection
        x = F.relu(x)
        return x
```

<br>

### 5.4. Value head
After the residual blocks, the network splits into two final processing blocks which output the value estimate and action probabilities. In AlphaGo and AlphaGo Zero, the value head estimated and optimized the probability of winning from a given game state. To handle cases with draws such as chess and shogi, AlphaZero's value head instead predicts the expected game outcome (win, lose, draw). The value estimate is used in the search process as an approximate value for the state, helping to guide the search to higher value states. 

Here is the structure of AlphaZero's value head:
- input &rarr; 1 convolutional filter (1x1) &rarr; batch norm &rarr; ReLU &rarr; fully connected layer &rarr; ReLU &rarr; fully connected layer &rarr; tanh output activation

Implementation in pytorch (different dimensions):

```python

class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False) 
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(in_features=1*6*7, out_features=3*6*7, bias=True)
        self.fc2 = nn.Linear(in_features=3*6*7, out_features=1, bias=True)
    
    def forward(self, x):
        v = F.relu(self.bn1(self.conv1(x)))
        v = v.view(-1, 6*7)
        v = F.relu(self.fc1(v))
        v = self.fc2(v)
        v = torch.tanh(v)
        return v  
```

<br>

### 5.5. Policy head
The policy head outputs action probabilities for each action given a game state. These action probabilities help guide the MCTS search. The search process improves on the initial prior probabilities received from the network.

Here is the structure of AlphaZero's policy head:
- input &rarr; 2 convolutional filters (2x2) &rarr; batch norm &rarr; ReLU &rarr; fully connected layer &rarr; output probabilities for each move

Implementation in pytorch (different dimensions):

``` python
class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.fc1 = nn.Linear(in_features=2*6*7, out_features=7)
    
    def forward(self, x):
        p = F.relu(self.bn1(self.conv1(x)))
        p = p.view(-1, 2*6*7)
        p = F.log_softmax(self.fc1(p), dim=1).exp()
        return p 
```

<br>

### 5.6. Putting it all together
Implementation in pytorch (different number of residual blocks):

```python
class AlphaNet(nn.Module):
    def __init__(self):
        super(AlphaNet, self).__init__()
        self.conv = ConvBlock()
        for i in range(10):
            setattr(self, f'res{i}', ResidualBlock())
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()
        
    def forward(self, x):
        x = self.conv(x)
        for i in range(10):
            x = getattr(self, f'res{i}')(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p
```

<br>

## 6. AlphaConnect
We've covered all the basics of AlphaZero, and I will now discuss the process of implementing my own version - AlphaConnect - to play connect4. 

Connect4 is a solved game and the first player is capable of a guaranteed win if they play correctly. Although AlphaConnect did not reach this optimal level of performance, over the course of 10k self-play games and learning steps AlphaConnect did reach a strong level of performance. I trained the system entirely on gpus freely available on google colab over the course of a couple weeks. Each colab instance consisted of 600 self-play games and learning steps, and ended with an evaluation phase comparing the latest version and the version from 600 steps ago. Each session of 600 self-play games and 200 evaluation games took about 8 hours, summing to slightly above 130 hours in total. 

 <!-- Anyone interested can play against the initial or final models by cloning the [github repository](https://github.com/ajpkim/alpha_connect4){:target="_blank"} and running ```play_initial_net``` or ```play_final_net```.  -->

### 6.1. Hyperparameters
The time and compute costs of searching for the best hyperparameters led me to use similar hyperparameters as AlphaZero with minor adjustments based on the scaled down nature of my project. I followed the AlphaZero learning rate schedule with step durations adjusted to match my hastened learning process. The learning rate started at 0.2 and was multiplied by 0.1 every 3,000 steps, ending at 0.0002. 

A hyperparameter I had to pick intuitively was the size of the replay buffer. AlphaGo Zero made use of a replay buffer that stored training examples from the previous 500,000 games. This is far beyond the scale of AlphaConnect! I opted for a two step approach to handling replay buffer size. The system begins with a small buffer that only holds 6,000 examples, roughly 300 games of data. After the initial 1,600 learning steps the buffer size is doubled to 12,000 examples, roughly 600 games of data. This setup allows the network to cycle through the early less informed training examples quickly and then take advantage of a larger buffer afterwards.

Given the limited training time and replay buffer size I choose to take advantage of the horizontal symmetry of connect4 and diversify the training data by randomly flipping training examples. Similar techniques were employed in AlphaZero's predecessors but notably removed in AlphaZero to make it a more general system. Board flipping is a way to get better learning results with limited resources and can be easily removed to extend AlphaConnect to different games. 

Here's the yaml configuration file used to train the network:

``` perl 
### Global params
game: Connect4
random_seed: 9
steps: 600
checkpoint_freq: 200

### MCTS params
n_simulations: 200
C_puct: 2.0
dirichlet_alpha: 0.75

### Eval params
eval: True
eval_sims: 100
eval_episodes: 100

### Training params
batch_size: 512
start_memory_capacity: 6000
end_memory_capacity: 12000
memory_step: 1600
lr: .2
lr_gamma: 0.1
lr_step_size: 3000
lr_scheduler_last_step: 11999
momentum: 0.9
weight_decay: 0.0001
horizontal_flip: True

```
<br>

### 6.2. Multithreading
The biggest weakness of AlphaConnect is that the system is single threaded and cannot make efficient use of gpu resources as a result. In contrast, AlphaZero uses a parallel MCTS algorithm in which multiple threads execute MCTS simulations at the same time. Parallelization means requests to the network can be batched together, games are completed faster, and computational resources are optimized. My single threaded version is simpler and less efficient. If I needed to scale AlphaConnect and train for much longer then it would be necessary to implement multithreading. Multithreading MCTS requires some care and uses virtual losses, temporary losses assigned to edges being explored by a thread, to circumvent the deterministic nature of the MCTS algorithm and allow the threads to effectively explore the game tree. More details can be found [here][multi-threaded MCTS]{:target="_blank"}.

[multi-threaded MCTS]: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf

<br>

## 7. Looking Ahead
Working on this project was a fun way to learn more about how these game playing deep reinforcement learning systems work. Games like Go, chess, and connect4 are well-defined and the systems I discussed (AlphaZero, AlphaConnect) are given perfect models of their world via the rules of the game. These domains are a far cry from the complexity of the outside world and from situations in which an agent must also learn the dynamics of their environment in addition to a good policy (as MuZero does). Still, games can be difficult, and mastering them requires some powerful forms of intelligence. Learning solely from experience in these game environments is an important step in being able to learn in more complex environments and solve less well-defined tasks.

Reinforcement learning as a general paradigm for artificial intelligence makes interesting contact with areas in cognitive science that have interested me for a long time. The focus on learning from experience and interacting in a world intuitively aligns RL with perspectives on intelligence that emphasize the role of adaptive learning in the emergence of intelligence. I think RL offers cool ways to study the principles governing the emergence of things like cooperation and intuitive physics. Doing so requires training agents in more complex environments that offer greater feedback and affordances.

Next, I'm looking forward to messing around with generative models for reinforcement learning or revisiting an old project - [Boids]{:target="_blank"} - and trying to achieve swarm like behavior with RL agents navigating predator/prey dynamics!

[Boids]: https://www.red3d.com/cwr/boids/

<br><br><br><br><br>
