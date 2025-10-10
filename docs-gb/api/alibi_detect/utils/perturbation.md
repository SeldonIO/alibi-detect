# `alibi_detect.utils.perturbation`
## Functions
### `apply_mask`

```python
apply_mask(X: numpy.ndarray, mask_size: tuple = (4, 4), n_masks: int = 1, coord: Optional[tuple] = None, channels: list = [0, 1, 2], mask_type: str = 'uniform', noise_distr: tuple = (0, 1), noise_rng: tuple = (0, 1), clip_rng: tuple = (0, 1)) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Mask images. Can zero out image patches or add normal or uniformly distributed noise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Batch of instances to be masked. |
| `mask_size` | `tuple` | `(4, 4)` | Tuple with the size of the mask. |
| `n_masks` | `int` | `1` | Number of masks applied for each instance in the batch X. |
| `coord` | `Optional[tuple]` | `None` | Upper left (x,y)-coordinates for the mask. |
| `channels` | `list` | `[0, 1, 2]` | Channels of the image to apply the mask to. |
| `mask_type` | `str` | `'uniform'` | Type of mask. One of 'uniform', 'random' (both additive noise) or 'zero' (zero values for mask). |
| `noise_distr` | `tuple` | `(0, 1)` | Mean and standard deviation for noise of 'random' mask type. |
| `noise_rng` | `tuple` | `(0, 1)` | Min and max value for noise of 'uniform' type. |
| `clip_rng` | `tuple` | `(0, 1)` | Min and max values for the masked instances. |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

### `brightness`

```python
brightness(x: numpy.ndarray, strength: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Change brightness of image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `strength` | `float` |  | Strength of brightness change. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `clipped_zoom`

```python
clipped_zoom(x: numpy.ndarray, zoom_factor: float) -> numpy.ndarray
```

Helper function for zoom blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `zoom_factor` | `float` |  | Zoom strength. |

**Returns**
- Type: `numpy.ndarray`

### `contrast`

```python
contrast(x: numpy.ndarray, strength: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Change contrast of image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `strength` | `float` |  | Strength of contrast change. Lower is actually more contrast. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `defocus_blur`

```python
defocus_blur(x: numpy.ndarray, radius: int, alias_blur: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply defocus blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `radius` | `int` |  | Radius for the Gaussian kernel. |
| `alias_blur` | `float` |  | Standard deviation for the Gaussian kernel in both X and Y directions. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `disk`

```python
disk(radius: float, alias_blur: float = 0.1, dtype = <class 'numpy.float32'>) -> numpy.ndarray
```

Helper function for defocus blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `radius` | `float` |  | Radius for the Gaussian kernel. |
| `alias_blur` | `float` | `0.1` | Standard deviation for the Gaussian kernel in both X and Y directions. |
| `dtype` |  | `<class 'numpy.float32'>` | Data type. |

**Returns**
- Type: `numpy.ndarray`

### `elastic_transform`

```python
elastic_transform(x: numpy.ndarray, mult_dxdy: float, sigma: float, rnd_rng: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply elastic transformation to instance.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `mult_dxdy` | `float` |  | Multiplier for the Gaussian noise in x and y directions. |
| `sigma` | `float` |  | Standard deviation determining the strength of the Gaussian perturbation. |
| `rnd_rng` | `float` |  | Range for random uniform noise. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `fog`

```python
fog(x: numpy.ndarray, fractal_mult: float, wibbledecay: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply fog to instance.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `fractal_mult` | `float` |  | Strength applied to `plasma_fractal` output. |
| `wibbledecay` | `float` |  | Decay factor for size of noise that is applied. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `gaussian_blur`

```python
gaussian_blur(x: numpy.ndarray, sigma: float, channel_axis: int = -1, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply Gaussian blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `sigma` | `float` |  | Standard deviation determining the strength of the blur. |
| `channel_axis` | `int` | `-1` | Denotes the axis of the colour channel. If `None` the image is assumed to be grayscale. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `gaussian_noise`

```python
gaussian_noise(x: numpy.ndarray, stdev: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Inject Gaussian noise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `stdev` | `float` |  | Standard deviation of noise. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `glass_blur`

```python
glass_blur(x: numpy.ndarray, sigma: float, max_delta: int, iterations: int, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply glass blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `sigma` | `float` |  | Standard deviation determining the strength of the Gaussian perturbation. |
| `max_delta` | `int` |  | Maximum pixel range for the blurring. |
| `iterations` | `int` |  | Number of blurring iterations. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `impulse_noise`

```python
impulse_noise(x: numpy.ndarray, amount: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Inject salt & pepper noise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `amount` | `float` |  | Proportion of pixels to replace with noise. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `inject_outlier_categorical`

```python
inject_outlier_categorical(X: numpy.ndarray, cols: List[int], perc_outlier: int, y: Optional[numpy.ndarray] = None, cat_perturb: Optional[dict] = None, X_fit: Optional[numpy.ndarray] = None, disc_perc: list = [25, 50, 75], smooth: float = 1.0) -> alibi_detect.utils.data.Bunch
```

Inject outliers in categorical variables of tabular data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Tabular data with categorical variables to perturb (inject outliers). |
| `cols` | `List[int]` |  | Columns of X that are categorical and can be perturbed. |
| `perc_outlier` | `int` |  | Percentage of observations which are perturbed to outliers. For multiple numerical features, the percentage is evenly split across the features. |
| `y` | `Optional[numpy.ndarray]` | `None` | Outlier labels. |
| `cat_perturb` | `Optional[dict]` | `None` | Dictionary mapping each category in the categorical variables to their furthest neighbour. |
| `X_fit` | `Optional[numpy.ndarray]` | `None` | Optional data used to infer pairwise distances from. |
| `disc_perc` | `list` | `[25, 50, 75]` | List with percentiles used in binning of numerical features used for the 'abdm' pairwise distance measure. |
| `smooth` | `float` | `1.0` | Smoothing exponent between 0 and 1 for the distances. Lower values will smooth the difference in distance metric between different features. |

**Returns**
- Type: `alibi_detect.utils.data.Bunch`

### `inject_outlier_tabular`

```python
inject_outlier_tabular(X: numpy.ndarray, cols: List[int], perc_outlier: int, y: Optional[numpy.ndarray] = None, n_std: float = 2.0, min_std: float = 1.0) -> alibi_detect.utils.data.Bunch
```

Inject outliers in numerical tabular data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Tabular data to perturb (inject outliers). |
| `cols` | `List[int]` |  | Columns of X that are numerical and can be perturbed. |
| `perc_outlier` | `int` |  | Percentage of observations which are perturbed to outliers. For multiple numerical features, the percentage is evenly split across the features. |
| `y` | `Optional[numpy.ndarray]` | `None` | Outlier labels. |
| `n_std` | `float` | `2.0` | Number of feature-wise standard deviations used to perturb the original data. |
| `min_std` | `float` | `1.0` | Minimum number of standard deviations away from the current observation. This is included because of the stochastic nature of the perturbation which could lead to minimal perturbations without a floor. |

**Returns**
- Type: `alibi_detect.utils.data.Bunch`

### `inject_outlier_ts`

```python
inject_outlier_ts(X: numpy.ndarray, perc_outlier: int, perc_window: int = 10, n_std: float = 2.0, min_std: float = 1.0) -> alibi_detect.utils.data.Bunch
```

Inject outliers in both univariate and multivariate time series data.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Time series data to perturb (inject outliers). |
| `perc_outlier` | `int` |  | Percentage of observations which are perturbed to outliers. For multivariate data, the percentage is evenly split across the individual time series. |
| `perc_window` | `int` | `10` | Percentage of the observations used to compute the standard deviation used in the perturbation. |
| `n_std` | `float` | `2.0` | Number of standard deviations in the window used to perturb the original data. |
| `min_std` | `float` | `1.0` | Minimum number of standard deviations away from the current observation. This is included because of the stochastic nature of the perturbation which could lead to minimal perturbations without a floor. |

**Returns**
- Type: `alibi_detect.utils.data.Bunch`

### `jpeg_compression`

```python
jpeg_compression(x: numpy.ndarray, strength: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Simulate changes due to JPEG compression for an image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `strength` | `float` |  | Strength of compression (>1). Lower is actually more compressed. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `pixelate`

```python
pixelate(x: numpy.ndarray, strength: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Change coarseness of pixels for an image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `strength` | `float` |  | Strength of pixelation (<1). Lower is actually more pixelated. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `plasma_fractal`

```python
plasma_fractal(mapsize: int = 256, wibbledecay: float = 3.0) -> numpy.ndarray
```

Helper function to apply fog to instance.

Generates a heightmap using diamond-square algorithm.
Returns a square 2d array, side length 'mapsize', of floats in range 0-255.
'mapsize' must be a power of two.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `mapsize` | `int` | `256` |  |
| `wibbledecay` | `float` | `3.0` |  |

**Returns**
- Type: `numpy.ndarray`

### `saturate`

```python
saturate(x: numpy.ndarray, strength: tuple, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Change colour saturation of image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `strength` | `tuple` |  | Strength of saturation change. Tuple consists of (multiplier, shift) of the perturbation. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `scale_minmax`

```python
scale_minmax(x: numpy.ndarray, xrange: Optional[tuple] = None) -> Tuple[numpy.ndarray, bool]
```

Minmax scaling to [0,1].

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Numpy array to be scaled. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `Tuple[numpy.ndarray, bool]`

### `shot_noise`

```python
shot_noise(x: numpy.ndarray, lam: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Inject Poisson noise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `lam` | `float` |  | Scalar for the lambda parameter determining the expectation of the interval. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `speckle_noise`

```python
speckle_noise(x: numpy.ndarray, stdev: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Inject speckle noise.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `stdev` | `float` |  | Standard deviation of noise. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`

### `zoom_blur`

```python
zoom_blur(x: numpy.ndarray, max_zoom: float, step_zoom: float, xrange: Optional[tuple] = None) -> numpy.ndarray
```

Apply zoom blur.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `numpy.ndarray` |  | Instance to be perturbed. |
| `max_zoom` | `float` |  | Max zoom strength. |
| `step_zoom` | `float` |  | Step size to go from 1 to `max_zoom` strength. |
| `xrange` | `Optional[tuple]` | `None` | Tuple with min and max data range. |

**Returns**
- Type: `numpy.ndarray`
