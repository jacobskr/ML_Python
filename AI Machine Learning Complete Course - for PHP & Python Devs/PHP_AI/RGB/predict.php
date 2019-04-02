<?php
require_once __DIR__ . '/../vendor/autoload.php';

use Phpml\ModelManager;

$newColorToIdentify = [244,149,65];

$filepath='network.ini';
$modelManager = new ModelManager();
$restoredClassifier = $modelManager->restoreFromFile($filepath);

$result = $restoredClassifier->predict($newColorToIdentify);
print_r($result);
