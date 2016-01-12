public class Network {
	private int numInputs;
	private int numOutputs;
	private int numHiddenLayers;
	private int neuronsPerHiddenLayer;
	
	double[][] errorArray;
	private NeuronLayer[] layers;
	
	public Network(int inputs, int outputs, int layers, int neurons) {
		numInputs = inputs;
		numOutputs = outputs;
		numHiddenLayers = layers;
		neuronsPerHiddenLayer = neurons;
		
		errorArray = new double[numHiddenLayers+1][];
		
		this.layers = new NeuronLayer[layers+1];
		for (int i=0; i< numHiddenLayers; i++) {
			this.layers[i] = new NeuronLayer(neuronsPerHiddenLayer, numInputs);
		}
		this.layers[numHiddenLayers] = new NeuronLayer(numOutputs, neuronsPerHiddenLayer);
	}
	//Weights: weights[layer][neuron][weight]
	public double[][][] getWeights() {
		double[][][] weights = new double[layers.length][][];
		for (int i=0; i<layers.length; i++) {
			weights[i] = new double[layers[i].numNeurons][];
			for (int j=0; j<layers[i].neuronLayer.length; j++) {
				weights[i][j] = new double[layers[i].neuronLayer[j].weights.length];
				for (int k=0; k<layers[i].neuronLayer[j].weights.length; k++) {
					weights[i][j][k] = layers[i].neuronLayer[j].weights[k];
				}
			}
		}
		//System.out.println(weights[2].length);
		return weights;
	}
	
	public int getNumberOfWeights() {
		int sum = numInputs * layers[0].numNeurons;
		for (int i=0; i<layers.length-1; i++) {
			sum+=layers[i].numNeurons * layers[i+1].numNeurons;
		}
		return sum;
	}
	
	public void putWeights(double[][][] weights) {
		for (int i=0; i<layers.length; i++) {
			for (int j=0; j<layers[i].neuronLayer.length; j++) {
				for (int k=0; k<layers[i].neuronLayer[j].weights.length; k++) {
					layers[i].neuronLayer[j].weights[k] = weights[i][j][k];
				}
			}
		}
	}
	
	public double[][] feedforward(double[] inputs) {
		double[] outputs = new double[layers[0].numNeurons];
		double[][] layerOutputs = new double[layers.length][];
		int weight = 0;
		
		if (inputs.length != numInputs) {
			layerOutputs=null;
			return layerOutputs;
		}
		
		for (int i=0; i<numHiddenLayers+1; i++) {
			layerOutputs[i] = new double[layers[i].neuronLayer.length];
			if (i>0) inputs = outputs;
			outputs = new double[layers[i].numNeurons];
			weight = 0;
			for (int j=0; j < layers[i].numNeurons; j++) {
				double sum = 0;
				int numinp = layers[i].neuronLayer[j].numInputs;
				for (int k=0; k<numinp; k++) {
					sum+= layers[i].neuronLayer[j].weights[k] * inputs[weight++];
				}
				sum+= layers[i].neuronLayer[j].weights[numinp-1] * -1;
				outputs[j] = sigmoid(sum);
				
				weight = 0;
			}
			layerOutputs[i]=outputs;
		}
		return layerOutputs;
	}
	
	public double sigmoid(double activation) {
		return 1.0/(1.0+Math.pow(Math.E, activation*-1));
	}
	
	public double[] calcError(double[] outputs, double[] targets) {
		double[] errors = new double[outputs.length];
		if (outputs.length!= targets.length) return null;
		for (int i=0; i<outputs.length; i++) {
			errors[i] = .5*(Math.pow(targets[i] - outputs[i],2));
		}
		return errors;
	}
	
	public double calcTotalError(double[] errors) {
		double sum=0;
		for(double n: errors) sum+=n;
		return sum;
	}
	
	//Weights: weights[layer][neuron][weight]
	public double[][] backprop(double[] input, double[] desired, double LR) {
		double[][][] weights = getWeights();
		double[][][] nonAlteredWeights = getWeights();
		double[][] layerOutputs = feedforward(input);
		
		errorArray[numHiddenLayers] = new double[layers[numHiddenLayers].numNeurons];
		
		for (int i=0; i<layers[numHiddenLayers].numNeurons; i++) {
			double actOut = layerOutputs[numHiddenLayers][i];
			double error = actOut* (1-actOut) * (desired[i]-actOut);
			errorArray[numHiddenLayers][i] = error;
			
			for (int j=0; j < layers[numHiddenLayers].neuronLayer[i].numInputs; j++) {
				double weight = weights[numHiddenLayers][i][j];
				double hiddenInput = layerOutputs[numHiddenLayers-1][i];
				double newWeight = weight + LR*error*hiddenInput;
				weights[numHiddenLayers][i][j] = newWeight;
			}
		}
		
		for (int i=(numHiddenLayers-1); i>0; i--) {
			for (int j=0; j<layers[i].neuronLayer.length; j++) {
				for (int k=0; k<layers[i].neuronLayer[j].weights.length; k++) {
					double weight = weights[i][j][k];
							//double priorWeight=0;
				
						
					double hiddenInput = layerOutputs[i-1][j];
					double actOut = layerOutputs[i][j];
					double sum = 0;
					for (int l=0; l<layers[i+1].numNeurons; l++) {
						
							//double priorWeight = nonAlteredWeights[i+1][l][k];
							sum+=errorArray[i+1][l]*weight;
						
					}
					double error = actOut* (1-actOut) * sum   ;
					double newWeight = weight + LR*error*hiddenInput;
					weights[i][j][k] = newWeight;
				}
			}
		}
		
		putWeights(weights);
		
		return feedforward(input);
		
	}
	
	public static double calcError(double[][] outputs, double[] targets,int numLayers) {
		double sum=0;
		for (int i=0; i<outputs[numLayers].length;i++) {
			sum+=.5*Math.pow(targets[i]-outputs[numLayers][i], 2);
		}
		return sum;
	}
	
	public static void main(String[] args) {
		
		int inputCount = 1;
		int outputCount = 1;
		int hiddenlayers = 1;
		int neurons = 7;
		Network test = new Network(inputCount,outputCount,hiddenlayers,neurons);
		double[][] inputs = {{0.7}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.1}, {0.8}, {0.9}};
	
		double[][] targets = {{0.99},{0.01},{0.01},{0.01},{0.01},{0.99},{0.01},{0.99},{0.99}};
		
		double lr = 0.3;
		double[][] outputs = null;
		double error=0.0;
		for (int j=0;j<inputs.length;j++) {
			for (int i=0;i<100000;i++){
				outputs = test.backprop(inputs[j], targets[j], lr);
				error = calcError(outputs, targets[j], hiddenlayers);
			}
		}	
		
		double[][][] finalWeights = test.getWeights();
		double[] testinput = {0.5};
		
		System.out.println(test.feedforward(testinput)[1][0]);
		
		


	}
	
}