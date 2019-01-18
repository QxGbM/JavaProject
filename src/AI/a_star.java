package AI;

import java.util.ArrayList;
import java.util.Random;

public class a_star {
	
	public static final int maxNodes = 100000;
	public static final double stepSize = 1e-3;
	
	public static final class State {
		
		public int moves;
		
		public double r1;
		public double r2;
		public double r3;
		
		public double z;
		
		public State (double r1, double r2, double r3, int moves) {
			this.r1 = r1;
			this.r2 = r2;
			this.r3 = r3;
			this.moves = moves;
			
			double t1 = 15 * (1 - r1);
			double t2 = 12 * (1 - r2);
			double t3 = 12 * (1 - r3);
			double t4 = 240 * r1;
			double t5 = 27 * r2;
			double t6 = 27 * r3;
			
			z = Math.max(t1, t2);
			z = Math.max(z, t3);
			z = Math.max(z, t4);
			z = Math.max(z, t5);
			z = Math.max(z, t6);
		}
		
		public double getH() {
			return z;
		}
		
		public int getMoves() {
			return moves;
		}
		
		public boolean equals(State s) {
			return Math.abs(this.r1 - s.r1) < stepSize && Math.abs(this.r2 - s.r2) < stepSize && Math.abs(this.r3 - s.r3) < stepSize;
		}
		
		public ArrayList<State> expand() {
			ArrayList<State> e = new ArrayList<State>();
			
			if (r1 + stepSize <= 1) {
				if (r2 - stepSize >= 0) {
					e.add(new State(r1 + stepSize, r2 - stepSize, r3, moves + 1));
				}
				if (r3 - stepSize >= 0) {
					e.add(new State(r1 + stepSize, r2, r3 - stepSize, moves + 1));
				}
			}
			
			if (r2 + stepSize <= 1) {
				if (r1 - stepSize >= 0) {
					e.add(new State(r1 - stepSize, r2 + stepSize, r3, moves + 1));
				}
				if (r3 - stepSize >= 0) {
					e.add(new State(r1, r2 + stepSize, r3 - stepSize, moves + 1));
				}
			}
			
			if (r3 + stepSize <= 1) {
				if (r1 - stepSize >= 0) {
					e.add(new State(r1 - stepSize, r2, r3 + stepSize, moves + 1));
				}
				if (r2 - stepSize >= 0) {
					e.add(new State(r1, r2 - stepSize, r3 + stepSize, moves + 1));
				}
			} 
			
			/*
			
			if (r1 + 2 * stepSize <= 1) {
				e.add(new State(r1 + 2 * stepSize, r2 - stepSize, r3 - stepSize, moves + 1));
			}
			
			if (r2 + stepSize <= 1) {
				e.add(new State(r1 - 2 * stepSize, r2 + stepSize, r3 + stepSize, moves + 1));
			}
			
			*/
			
			return e;
		}
	}
	
	public static void main (String[] args) {
		
		Random r = new Random();
		ArrayList<State> states = new ArrayList<State>();
		
		double r1 = r.nextDouble() / 2, r2 = r.nextDouble() / 2, r3 = 1 - r1 - r2;
		
		//double r1 = r.nextDouble(), r2 = (1 - r1) / 2, r3 = r2;
		
		states.add(new State(r1, r2, r3, 0));
		State minS = states.get(0);
		
		for (int i = 0, j = 1; i < maxNodes && i < j; i++) {
			if (states.get(i).getH() < minS.getH()) {
				minS = states.get(i);
			}
			
			ArrayList<State> expanded = states.get(i).expand();
			
			while (!expanded.isEmpty()) {
				int k;
				State s = expanded.remove(0);
				
				for (k = 0; k < j; k++) {
					if (s.equals(states.get(k))) {
						if(s.getMoves() < states.get(k).getMoves()) 
						{states.remove(k); if(k < i) i--; j--;}
						else break;
					}
				}
				
				if (k == j) {
					double f = s.getH() + s.getMoves();
					for (k = i + 1; k < j && states.get(k).getH() + states.get(k).getMoves() <= f; k++) {}
					states.add(k, s); j++;
				}
			}
		}
		
		System.out.println("r1: " + minS.r1);
		System.out.println("r2: " + minS.r2);
		System.out.println("r3: " + minS.r3);
		
		System.out.println("z: " + minS.z);
		
	}
}
