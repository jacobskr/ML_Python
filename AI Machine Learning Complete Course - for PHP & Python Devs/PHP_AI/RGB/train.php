<?php
require_once __DIR__ . '/../vendor/autoload.php';

use Phpml\Metric\Accuracy;
use Phpml\NeuralNetwork\Layer;
use Phpml\NeuralNetwork\Node\Neuron;
use Phpml\Classification\MLPClassifier;
use Phpml\NeuralNetwork\ActivationFunction\PReLU;
use Phpml\NeuralNetwork\ActivationFunction\Sigmoid;
use Phpml\ModelManager;

$layer1 = new Layer(3, Neuron::class, new PReLU);
$layer2 = new Layer(3, Neuron::class, new Sigmoid);
$mlp = new MLPClassifier(3, [$layer1, $layer2], ['red', 'blue', 'green'], 30000, null, 0.01);

$mlp->train(
    $samples = [
        [0, 0, 255],
        [0, 0, 192],
        [208, 0, 49],
        [228,  105, 116],
        [128, 80, 255],
        [248,  80, 68],
        [124,252,0],
        [127,255,0],
        [50,205,50],
        [34,139,34],
        [0,100,0],
        [107,142,35],
        [220,20,60],
        [178,34,34],
        [255,0,0]
    ],
    $targets = [
        'blue',
        'blue',
        'red',
        'red',
        'blue',
        'red',
        'green',
        'green',
        'green',
        'green',
        'green',
        'green',
        'red',
        'red',
        'red'
    ]
);

$filepath = 'network.ini';
$modelManager = new ModelManager();
$modelManager->saveToFile($mlp, $filepath);

$predictedLabels = $mlp->predict($samples);
print_r('Accuracy: '.Accuracy::score($targets, $predictedLabels));
