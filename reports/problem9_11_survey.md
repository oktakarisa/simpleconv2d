# Problem 9 & 11: Survey of Image Recognition Models and Filter Sizes

## Problem 9: Survey of Famous Image Recognition Models

### AlexNet (2012)

AlexNet was the breakthrough CNN that won the ImageNet competition in 2012 by a huge margin. It showed that deep neural networks trained on GPUs could significantly outperform traditional methods.

**Architecture:**
- 8 layers total (5 convolutional + 3 fully connected)
- Used ReLU activation instead of tanh/sigmoid
- Introduced dropout for regularization
- Used overlapping max pooling
- Trained on two GPUs due to memory limitations

**Key innovations:**
- First to use ReLU at scale (faster training than sigmoid)
- Local Response Normalization (LRN)
- Data augmentation techniques
- GPU implementation for training

**Typical layer structure:**
1. Conv 11×11, stride 4, 96 filters
2. MaxPool 3×3, stride 2
3. Conv 5×5, 256 filters
4. MaxPool 3×3, stride 2
5. Conv 3×3, 384 filters
6. Conv 3×3, 384 filters
7. Conv 3×3, 256 filters
8. MaxPool 3×3, stride 2
9. FC 4096 nodes
10. FC 4096 nodes
11. FC 1000 nodes (output)

**Parameters:** ~60 million

**Why it matters:**
AlexNet proved that deep learning works for image recognition and started the modern deep learning revolution. It's still studied today as a foundational architecture.

---

### VGG16 (2014)

VGG16, developed by the Visual Geometry Group at Oxford, demonstrated that network depth is crucial for performance. It's known for its simplicity and uniform architecture.

**Architecture:**
- 16 layers with learnable weights (13 conv + 3 FC)
- All convolution filters are 3×3 with stride 1
- All max pooling is 2×2 with stride 2
- Channels double after each pooling: 64 → 128 → 256 → 512 → 512

**Key innovations:**
- Consistent use of small 3×3 filters throughout
- Showed that depth matters more than filter size
- Simple and homogeneous architecture (easy to understand and implement)
- Better than using larger filters like 5×5 or 7×7

**Typical layer structure:**
1. Conv3-64 × 2, MaxPool
2. Conv3-128 × 2, MaxPool  
3. Conv3-256 × 3, MaxPool
4. Conv3-512 × 3, MaxPool
5. Conv3-512 × 3, MaxPool
6. FC 4096
7. FC 4096
8. FC 1000

(Conv3-64 means 3×3 convolution with 64 filters)

**Parameters:** ~138 million

**Why it matters:**
VGG showed that a simple, deep architecture with small filters works better than shallow networks with large filters. It's still used as a backbone in many applications today.

**Variants:**
- VGG11, VGG13, VGG16, VGG19 (different depths)
- VGG16 and VGG19 are most commonly used

---

### Other Notable Architectures (Brief Overview)

**ResNet (2015)**
- Introduced skip connections (residual connections)
- Enabled training of very deep networks (50, 101, 152 layers)
- Won ImageNet 2015

**Inception/GoogLeNet (2014)**
- Used "inception modules" with multiple filter sizes in parallel
- 1×1 convolutions for dimensionality reduction
- Much fewer parameters than VGG

**MobileNet (2017)**
- Designed for mobile/embedded devices
- Uses depthwise separable convolutions
- Much faster and lighter than standard CNNs

**EfficientNet (2019)**
- Systematically scales depth, width, and resolution
- State-of-the-art performance with fewer parameters

---

## Problem 11: Survey on Filter Sizes

### Why 3×3 Filters Are Commonly Used

The use of 3×3 filters has become standard in modern CNN architectures. Here's why:

#### 1. Smaller Filters, More Depth

Two stacked 3×3 convolution layers have an effective receptive field of 5×5, and three stacked 3×3 layers have a receptive field of 7×7. However, using multiple small filters has advantages:

**Comparison: One 7×7 vs Three 3×3 filters**

*Parameters (assuming C channels):*
- One 7×7 filter: C × C × 7 × 7 = 49C² parameters
- Three 3×3 filters: 3 × (C × C × 3 × 3) = 27C² parameters

The three 3×3 approach uses **~45% fewer parameters** while achieving the same receptive field!

#### 2. More Non-linearity

Each convolution layer is followed by an activation function (like ReLU). Using three 3×3 filters means you get three non-linear activations instead of just one with a 7×7 filter. This allows the network to learn more complex functions.

#### 3. Better Feature Learning

Smaller filters force the network to learn more hierarchical features:
- First 3×3 layer: learns simple edges
- Second 3×3 layer: combines edges into simple shapes
- Third 3×3 layer: combines shapes into complex patterns

This hierarchical learning is more powerful than trying to learn everything in one large filter.

#### 4. Computational Efficiency

Smaller filters are more efficient to compute:
- Better memory access patterns
- More opportunities for optimization
- Can use more efficient hardware implementations

#### 5. Historical Validation

VGG demonstrated empirically that using small 3×3 filters throughout the network works better than using larger filters. This has been validated across countless architectures since then.

---

### The Effect of 1×1 Filters

At first glance, 1×1 filters seem pointless since they don't look at neighboring pixels. However, they're actually very useful:

#### 1. Dimensionality Reduction (Channel-wise)

A 1×1 filter operates across channels, not spatial dimensions.

**Example:**
- Input: 64 channels, 28×28 spatial size
- Apply 32 1×1 filters
- Output: 32 channels, 28×28 spatial size

The spatial size stays the same, but we've reduced from 64 to 32 channels. This is called a "bottleneck" and reduces computational cost.

#### 2. Adding Non-linearity

Even though a 1×1 conv doesn't look at neighbors, it's followed by an activation function. This adds non-linearity without changing spatial dimensions, letting the network learn more complex channel relationships.

#### 3. Cross-Channel Information

A 1×1 filter looks at all channels at each pixel position and creates new channel combinations. Think of it as a fully connected layer applied to each pixel position independently.

**Mathematical view:**
- Input: X with shape (H, W, C_in)
- 1×1 conv with C_out filters
- Output: Y with shape (H, W, C_out)
- At each (i,j) position: Y[i,j,:] = W × X[i,j,:] + b

This is equivalent to a fully connected layer across channels.

#### 4. Practical Applications

**In Inception Networks:**
- 1×1 convs reduce channels before expensive 3×3 and 5×5 convolutions
- Example: 256 channels → 64 channels (1×1) → 64 channels (3×3)
- This is much cheaper than 256 → 256 (3×3) directly

**In ResNet:**
- Bottleneck blocks use 1×1 to reduce and then restore dimensions
- Pattern: 256 → 64 (1×1) → 64 (3×3) → 256 (1×1)

**In MobileNet:**
- Used heavily in depthwise separable convolutions
- Separates spatial and channel-wise operations

#### 5. When NOT to Use 1×1

Since 1×1 doesn't look at neighbors, it can't learn spatial patterns. You always need some spatial convolutions (3×3, 5×5) in your network. 1×1 is a tool for efficiency and channel manipulation, not spatial feature learning.

---

## Summary

**3×3 Filters:**
- More parameter efficient than larger filters
- Enable deeper networks with more non-linearity
- Learn hierarchical features better
- Standard choice in modern architectures

**1×1 Filters:**
- Reduce/increase channel dimensions
- Add non-linearity without spatial operations
- Enable efficient "bottleneck" architectures
- Learn cross-channel correlations

Both filter sizes are complementary and are used together in most modern CNN architectures. The key is understanding when and why to use each one.

