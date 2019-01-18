package OperatingSystems;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

public class ParallelPuzzleSolver {
	
	public static Random RNG = new Random();
	public static int maxNodes = 100;

	public static void main(String[] args){
		long startTime = System.nanoTime();
		
		ParallelPuzzleSolver ps = new ParallelPuzzleSolver();
		Puzzle p = new Puzzle("");

		if (args.length > 0) try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(args[0])));
			while (reader.ready()) {
				args = reader.readLine().split("\\s+");
				ps.parseCommand(p, args);
			}
			reader.close();
			} catch (IOException e) {System.out.println("Input File Error");}
		else {
			Scanner input = new Scanner(System.in);
			args = input.nextLine().split("\\s+");
			while (!args[0].equals("")) {
				ps.parseCommand(p, args);
				args = input.nextLine().split("\\s+");
			}
			input.close();
		}
		
		long endTime = System.nanoTime();
		long totalTime = endTime - startTime;
		System.out.println(totalTime);
		
	}
	
	private void parseCommand(Puzzle p, String[] args){
		if (args[0].startsWith("//")) {
			return;
		}
		else if (args[0].equals("setState") && args.length >= 2) {
			String state = "";
			for (int i = 1; i < args.length; i++) state += args[i];
			p.setState(state);
		}
		else if (args[0].equals("randomizeState") && args.length == 2) {
			p.randomizeState(Integer.valueOf(args[1]));
		}
		else if (args[0].equals("printState") && args.length == 1) {
			p.printState();
		}
		else if (args[0].equals("move") && args.length == 2) {
			if (!p.move(args[1])) {System.out.println("Improper Command");}
		}
		else if (args[0].equals("solve") && args.length == 2) {
			p.resetMoves();
			System.out.println("Using local beam search k = " + args[1]);
			System.out.println(solve(p, Integer.valueOf(args[1])));
		}
		else if (args[0].equals("maxNodes") && args.length == 2) {
			maxNodes = Integer.valueOf(args[1]);
		}
		else System.out.println("Improper Command");
	}
	
	private LinkedList<Puzzle> states;
	private volatile int completedThreads;
	
	private class Randomizer extends Thread {
		
		private int i;
		private int k;
		private Puzzle p;
		private Object lock;
		
		public Randomizer (int i, int k, Puzzle p, Object lock) {
			super();
			this.i = i;
			this.k = k;
			this.p = p;
			this.lock = lock;
		}
		
		@Override
		public void run() {
			p.randomizeState(k);
			states.set(i, p);
			synchronized (lock) {
				completedThreads ++;
			}
		}
	}
	
	private class Expander extends Thread {
		
		private Puzzle p;
		private Object lock;
		
		public Expander (Puzzle p, Object lock) {
			super();
			this.p = p;
			this.lock = lock;
		}
		
		@Override
		public void run() {
			LinkedList<Puzzle> list = p.expand();
			synchronized (lock) {
				states.addAll(list);
				completedThreads ++;
			}
		}
	}
	
	
	private String solve (Puzzle p, int k) {
		states = new LinkedList<Puzzle>();
		Object lock = new Object();
		Puzzle.PuzzleComparator puzzlecomparator = new Puzzle.PuzzleComparator();
		
		completedThreads = 0;
		for (int i = 0; i < k; i++) {
			states.add(new Puzzle(""));
			new Randomizer(i, 10, p.clone(), lock).start();
		}
		
		while (completedThreads < k) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < maxNodes; i++) {
			completedThreads = 0;
			for (int j = 0; j < k; j++)
				new Expander(states.get(j).clone(), lock).start();
			while (completedThreads < k) {
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {e.printStackTrace();}
			}
			states.sort(puzzlecomparator);
			for (int j = 0; j < states.size() - 1; j++) {
				if (puzzlecomparator.compare(states.get(j), states.get(j+1)) == 0) {
					states.remove(j); j--;
				}
			}
				
			while (states.size() > k)
				states.remove(k);
			if (states.get(0).isGoal()) 
				return states.get(0).getMoves();
			
			if (states.size() < k) {
				completedThreads = states.size();
				for (int m = states.size(); m < k; m++) {
					states.add(new Puzzle(""));
					new Randomizer(m, 10, p.clone(), lock).start();
				}
				
				while (completedThreads < k) {
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
			
			System.out.println(i + ":");
			for(int j = 0; j < k; j++)
				System.out.println(states.get(j));
		}

		return "Searched States Exceeds Limit";
	}
}