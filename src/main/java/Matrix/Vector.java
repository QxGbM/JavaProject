package Matrix;

import java.util.Random;

public final class Vector {
	
	private int d;
	
	private double[] e;
	
	public double[] elements() {
		return e;
	}
	
	public int dimension() {
		return d;
	}
	
	public Vector (int d) {
		this.d = d;
		e = new double[d];
	}
	
	public Vector (double[] v) {
		d = v.length;
		e = new double[d];
		for (int i = 0; i < d; i++) {
			e[i] = v[i];
		}
	}
	
	public String toString() {
		String s = "[";
		for (int i = 0; i < d-1; i++) {
			s += Double.toString(e[i]) + ", ";
		}
		s += Double.toString(e[d-1]) + "]";
		return s;
	}

	public static Vector randomVector(int d, double min, double max) {
		Random r = new Random();
		Vector v = new Vector(d);
		for (int i = 0; i < d; i++) {
			v.e[i] = min + (max - min) * r.nextDouble();
		}
		return v;
	}
}
