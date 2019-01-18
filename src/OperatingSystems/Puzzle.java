package OperatingSystems;

import java.util.Comparator;
import java.util.LinkedList;

public class Puzzle {
	private int[] content = new int[16];
	private int l = 0;
	
	private int h;
	private String moves;
	
	public Puzzle (String moves) {
		this.moves = moves;
		h = 0;
		for (int i = 0; i < 16; i++)
			h += Math.abs(content[i]-i) / 4 + Math.abs(content[i]-i) % 4;
	}
	
	public int getH() {
		return h;
	}
	
	public String getMoves() {
		return moves;
	}
	
	public void resetMoves() {
		moves = "";
	}
	
	public void setState (String state) {
		h = 0;
		for (int i = 0; i < 16 ; i++) {
			if (state.charAt(i) == 'x') {l = i; content[i] = 0;}
			else if (state.charAt(i) >= '0' && state.charAt(i) <= '9') {content[i] = state.charAt(i) - '0';}
			else if (state.charAt(i) >= 'a' && state.charAt(i) <= 'f') {content[i] = state.charAt(i) - 'a' + 10;}
			h += Math.abs(content[i]-i) / 4 + Math.abs(content[i]-i) % 4;
		}
	}
	
	public boolean isGoal () {
		for (int i = 0; i < 16; i++) 
			if (content[i] != i) return false;
		return true;
	}
	
	public void randomizeState (int n) {
		for (int i = 1; i <= n;) 
			switch(ParallelPuzzleSolver.RNG.nextInt(4)) {
			case 0: if(moveUp()) {i++;} break;
			case 1: if(moveDown()) {i++;} break;
			case 2: if(moveLeft()) {i++;} break;
			case 3: if(moveRight()) {i++;} break;
			}
	}
	
	public void printState(){
		for (int i = 0; i < 16; i++) {
			if (i == l) System.out.print("x"); else System.out.print(content[i]);
			if (i % 4 == 3) System.out.println(); else System.out.print(" ");
		}
		System.out.println("h = " + h + " moves: " + moves + "\n");
	}
	
	public boolean move(String direction) {
		if (direction.equals("left")) {return moveLeft();}
		else if (direction.equals("right")) {return moveRight();}
		else if (direction.equals("up")) {return moveUp();}
		else if (direction.equals("down")) {return moveDown();}
		else return false;
	}
	
	private boolean moveUp() {
		if (l <= 3) return false;
		swap(l, l-4); l -= 4;
		if (moves.endsWith("d")) moves = moves.substring(0, moves.length() - 1);
		else moves += "u";
		return true;
	}
	private boolean moveDown() {
		if (l >= 12) return false;
		swap(l, l+4); l += 4;
		if (moves.endsWith("u")) moves = moves.substring(0, moves.length() - 1);
		else moves += "d";
		return true;
	}
	
	private boolean moveLeft() {
		if (l % 4 == 0) return false;
		swap(l, l-1); l -= 1;
		if (moves.endsWith("r")) moves = moves.substring(0, moves.length() - 1);
		else moves += "l";
		return true;
	}
	
	private boolean moveRight() {
		if (l % 4 == 3) return false;
		swap(l, l+1); l += 1;
		if (moves.endsWith("l")) moves = moves.substring(0, moves.length() - 1);
		else moves += "r";
		return true;
	}
	
	private void swap(int i, int j) {
		h -= Math.abs(content[i]-i) / 4 + Math.abs(content[i]-i) % 4;
		h -= Math.abs(content[j]-j) / 4 + Math.abs(content[j]-j) % 4;
		int t = content[i]; content[i] = content[j]; content[j] = t;
		h += Math.abs(content[i]-i) / 4 + Math.abs(content[i]-i) % 4;
		h += Math.abs(content[j]-j) / 4 + Math.abs(content[j]-j) % 4;
	}
	
	public Puzzle clone() {
		Puzzle p = new Puzzle(moves); 
		p.l = this.l;
		p.h = this.h;
		for (int i = 0; i < 16 ; i++) p.content[i] = this.content[i];
		return p;
	}
	
	public LinkedList<Puzzle> expand(){
		LinkedList<Puzzle> result = new LinkedList<Puzzle>();
		if (moveUp()) {result.add(clone()); moveDown();}
		if (moveDown()) {result.add(clone()); moveUp();}
		if (moveLeft()) {result.add(clone()); moveRight();}
		if (moveRight()) {result.add(clone()); moveLeft();}
		return result;
	}
	
	public String toString() {
		String s = "";
		for (int i = 0; i < 16; i++) s += content[i] + " ";
		return s + " " + moves + " " + h; 
	}
	
	public static class PuzzleComparator implements Comparator<Puzzle> {

		@Override
		public int compare(Puzzle p1, Puzzle p2) {
			if (p1.h < p2.h) return -1;
			else if (p1.h == p2.h) return 0;
			else return 1;
		}
		
	}
}
