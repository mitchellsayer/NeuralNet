import java.util.Random;

public class Neuron {
	
	public int numInputs;
	public double[] weights;
	
	public Neuron(int ni) {
		this.numInputs = ni;
		weights = new double[ni+1];
		Random r = new Random();
		for (int i=0; i < ni+1; i++) {
			double randNum = r.nextGaussian();
			weights[i] = randNum;
		}
	}
	
}
