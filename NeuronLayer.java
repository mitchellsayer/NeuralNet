
public class NeuronLayer {
	
	public int numNeurons;
	public Neuron[] neuronLayer;
	
	public NeuronLayer(int numNeu, int inputsPerNeuron) {
		this.numNeurons = numNeu;
		neuronLayer = new Neuron[numNeu];
		for (int i=0; i<numNeu; i++) {
			neuronLayer[i] = new Neuron(inputsPerNeuron);
		}
	}
	
}
