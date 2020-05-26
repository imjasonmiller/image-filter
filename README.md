# image-filter ✨
A playground in which I experiment and learn about image filters.

## Usage

### Flags
 Flag         | Details                   | Default 
--------------|---------------------------|------------
-i / --input  | Image input               | input.jpg
-o / --output | Image output              | output.jpg
-x            | Crop x-coordinate         | 0
-y            | Crop y-coordinate         | 0
-w / --width  | Crop width                | Image width
-h / --height | Crop height               | Image width

An explanation for each setting can also be found via:
```shell
$ image-filter --help
```

### Subcommands

### Gaussian blur

 Flag        | Details              | Default
-------------|----------------------|-----------
-s / --sigma | Strength of the blur | 0.84089642

```shell
$ image-filter -i a.jpg -o b.jpg gaussian1d -s 10.0
```

```shell
$ image-filter -i a.jpg -o b.jpg gaussian2d -s 10.0
```

Image, ["Chaotic Clouds of Jupiter"](https://www.jpl.nasa.gov/spaceimages/details.php?id=PIA22424), via NASA/JPL-Caltech/SwRI/MSSS/Gerald Eichstad/Sean Doran @ is CC-NC-SA

