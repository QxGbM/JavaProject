package AI;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class PuzzleSolver{
	
	static int maxNodes = 100;

	public static void main(String[] args){
		Puzzle p = new Puzzle8();
		if (args.length > 0 && args[args.length - 1].equals("R")) p = new PuzzleR();
		if (args.length > 0 && !args[0].equals("R")) try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(args[0])));
			while (reader.ready()) {
				args = reader.readLine().split("\\s+");
				parseCommand(p, args);
			}
			reader.close();
			} catch (IOException e) {System.out.println("Input File Error");}
		else {
			Scanner input = new Scanner(System.in);
			args = input.nextLine().split("\\s+");
			while (!args[0].equals("")) {
				parseCommand(p, args);
				args = input.nextLine().split("\\s+");
			}
			input.close();
		}
	}
	
	public static void parseCommand(Puzzle p, String[] args){
		if (args[0].equals("setState") && args.length >= 2) {
			String state = "";
			for (int i = 1; i < args.length; i++) state += args[i];
			p.setState(state);
		} else if (args[0].equals("randomizeState") && args.length == 2) {
			p.randomizeState(Integer.valueOf(args[1]));
		} else if (args[0].equals("printState") && args.length == 1) {
			p.printState();
		} else if (args[0].equals("move") && args.length == 2) {
			if (!p.move(args[1])) {System.out.println("Improper Command");}
		} else if (args[0].equals("solve") && args.length == 3) {
			if (args[1].equals("A-star")) 
			{System.out.println("A* heuristic:" + args[2] + " " + A_star(p, args[2]));}
			else if (args[1].equals("beam")) 
			{System.out.println("beam k = " + args[2] + " " + beam(p, Integer.valueOf(args[2])));}
			else System.out.println("Improper Command");
		} else if (args[0].equals("maxNodes") && args.length == 2) {
			maxNodes = Integer.valueOf(args[1]);
		} else System.out.println("Improper Command");
	}
	
	public static String A_star(Puzzle p, String heuristic){
		ArrayList<State> states = new ArrayList<State>();
		states.add(p.generateState(""));
		states.get(0).calculateH(heuristic);
		for (int i = 0, j = 1; i < maxNodes; i++){
			if (states.get(i).isGoal()) return states.get(i).getMoves();
			ArrayList<State> expanded = states.get(i).expand();
			while (!expanded.isEmpty()){
				int k;
				State s = expanded.remove(0);
				for (k = 0; k < j; k++) {
					if (s.equals(states.get(k))) {
						if(s.getMoves().length() < states.get(k).getMoves().length()) 
						{states.remove(k); if(k < i) i--; j--;}
						else break;
					}
				}
				if (k == j) {
					int f = s.calculateH(heuristic) + s.getMoves().length();
					for (k = i + 1; k < j && states.get(k).getH() + states.get(k).getMoves().length() <= f; k++) {}
					states.add(k, s); j++;
				}
			}
		}
		return "Searched States Exceeds Limit";
	}
	
	public static String beam(Puzzle p, int k){
		ArrayList<State> states = new ArrayList<State>();
		states.add(p.generateState(""));
		states.get(0).calculateH("h1");
		for (int i = 0, j = 1; i < maxNodes; i++){
			if (states.get(i).isGoal()) return states.get(i).getMoves();
			ArrayList<State> expanded = states.get(i).expand();
			while (!expanded.isEmpty()){
				int min = Integer.MAX_VALUE, t = 0, l = 0;
				for (l = 0; l < expanded.size(); l++) 
					if (expanded.get(l).calculateH("h1") < min) 
					{min = expanded.get(l).getH(); t = l;}
				State s = expanded.remove(t);
				for (l = 0; l < j && !s.equals(states.get(l)); l++) {}
				if (l == j) {
					for (l = i + 1; l < j && states.get(l).getH() <= min; l++) {}
					if (l <= i + k) {states.add(l, s); j++;}
				}
			}
		}
		return "Searched States Exceeds Limit";
	}
}

interface State {
	public Puzzle getP();
	public String getMoves();
	public int getH();
	public ArrayList<State> expand();
	public boolean isGoal();
	public int calculateH(String heuristic);
	public boolean equals(State s);
}
class State8 implements State {
	private Puzzle8 p;
	private String moves;
	private int h;
	
	public State8(Puzzle8 P, String MOVES)
	{p = P; moves = MOVES;}
	
	public Puzzle getP() { return p; }
	public String getMoves() {return moves;	}
	public int getH() { return h; }
	
	public ArrayList<State> expand(){
		ArrayList<State> result = new ArrayList<State>();
		if (p.move("up")) {result.add(p.clone().generateState(moves + "u")); p.move("down");}
		if (p.move("down")) {result.add(p.clone().generateState(moves + "d")); p.move("up");}
		if (p.move("left")) {result.add(p.clone().generateState(moves + "l")); p.move("right");}
		if (p.move("right")) {result.add(p.clone().generateState(moves + "r")); p.move("left");}
		return result;
	}
	
	public boolean isGoal() {
		return p.isInitial();
	}
	
	public int calculateH(String heuristic) {
		h = 0;
		if (heuristic.equals("h1")){
			for (int i = 0; i < 9; i++)	if (p.getContent()[i] != i) h++;
		}
		else if (heuristic.equals("h2")){
			for (int i = 0; i < 9; i++) 
			{h += Math.abs(p.getContent()[i]-i) / 3; h += Math.abs(p.getContent()[i]-i) % 3;}
		}
		else h = Integer.MAX_VALUE;
		return h;
	}
	
	public boolean equals(State s) {
		int[] t = s.getP().getContent();
		for (int i = 0; i < 9 ; i++) if (p.getContent()[i] != t[i]) return false;
		return true;
	}
}
class StateR implements State{
	private PuzzleR p;
	private String moves;
	private int h;

	public StateR(PuzzleR P, String MOVES)
	{p = P; moves = MOVES;}
	
	public Puzzle getP() { return p;}
	public String getMoves() { return moves;}
	public int getH() { return h;}
	
	public ArrayList<State> expand(){
		ArrayList<State> result = new ArrayList<State>();
		if (p.move("U")) {result.add(p.clone().generateState(moves + "U ")); p.move("U\'");}
		if (p.move("U2")) {result.add(p.clone().generateState(moves + "U2 ")); p.move("U2");}
		if (p.move("U\'")) {result.add(p.clone().generateState(moves + "U\' ")); p.move("U");}
		if (p.move("F")) {result.add(p.clone().generateState(moves + "F ")); p.move("F\'");}
		if (p.move("F2")) {result.add(p.clone().generateState(moves + "F2 ")); p.move("F2");}
		if (p.move("F\'")) {result.add(p.clone().generateState(moves + "F\' ")); p.move("F");}
		if (p.move("R")) {result.add(p.clone().generateState(moves + "R ")); p.move("R\'");}
		if (p.move("R2")) {result.add(p.clone().generateState(moves + "R2 ")); p.move("R2");}
		if (p.move("R\'")) {result.add(p.clone().generateState(moves + "R\' ")); p.move("R");}
		return result;
	}
	
	public boolean isGoal() {
		return p.isInitial();
	}
	
	public int calculateH(String heuristic) {
		h = 0;
		if (heuristic.equals("h1")){
			for (int i = 0; i < 24; i++) if (p.getContent()[i] != i / 4) h++;
		}
		else h = Integer.MAX_VALUE;
		return h;
	}
	
	public boolean equals(State s) {
		int[] t = s.getP().getContent();
		for (int i = 0; i < 24 ; i++) if (p.getContent()[i] != t[i]) return false;
		return true;
	}
}
interface Puzzle {
	public int[] getContent();
	public void setState(String state);
	public boolean isInitial();
	public void randomizeState(int n);
	public void printState();
	public boolean move(String direction);
	public Puzzle clone();
	public State generateState(String moves);
}
class Puzzle8 implements Puzzle{
	private int[] content = new int[9];
	private int l = 0;
	
	public Puzzle8(){}
	public int[] getContent() {return content;}
	
	public void setState (String state) {
		for (int i = 0; i < 9 ; i++){
			if (state.charAt(i) == 'b') {l = i; content[i] = 0;}
			else {content[i] = state.charAt(i) - '0';}
		}
	}
	
	public boolean isInitial () {
		for (int i = 0; i < 9; i++)	if (content[i] != i) return false;
		return true;
	}
	
	public void randomizeState (int n) {
		Random r = new Random("seed".hashCode());
		for (int i = 1; i <= n;) switch(r.nextInt(4)) {
			case 0: if(moveUp()) {i++;} break;
			case 1: if(moveDown()) {i++;} break;
			case 2: if(moveLeft()) {i++;} break;
			case 3: if(moveRight()) {i++;} break;
			}
	}
	
	public void printState(){
		for (int i = 0; i < 9; i++) {
			if (i == l) System.out.print("b"); else System.out.print(content[i]);
			if (i % 3 == 2) System.out.println(); else System.out.print(" ");
		}
	}
	public boolean move(String direction) {
		if (direction.equals("left")) {return moveLeft();}
		else if (direction.equals("right")) {return moveRight();}
		else if (direction.equals("up")) {return moveUp();}
		else if (direction.equals("down")) {return moveDown();}
		else return false;
	}
	private boolean moveUp() {
		if (l < 3) return false;
		int t = content[l]; content[l] = content[l-3]; content[l-3] = t; l -= 3; return true;
	}
	private boolean moveDown() {
		if (l >= 6) return false;
		int t = content[l]; content[l] = content[l+3]; content[l+3] = t; l += 3; return true;
	}
	private boolean moveLeft() {
		if (l % 3 == 0) return false;
		int t = content[l]; content[l] = content[l-1]; content[l-1] = t; l -= 1; return true;
	}
	private boolean moveRight() {
		if (l % 3 == 2) return false;
		int t = content[l]; content[l] = content[l+1]; content[l+1] = t; l += 1;  return true;
	}
	
	public Puzzle clone() {
		Puzzle8 p = new Puzzle8(); p.l = this.l;
		for (int i = 0; i < 9 ; i++) p.content[i] = this.content[i];
		return p;
	}
	
	public State generateState(String moves){
		return new State8(this, moves);
	}
}

class PuzzleR implements Puzzle {
	
	public int [] content = new int[24];
	
	private char [] colors = {'w','g','o','b','r','y'};
	
	private int[] Up = {0,1,2,3,13,12,9,8,5,4,17,16};
	private int[] Front = {4,5,6,7,2,3,8,10,20,21,19,17};
	private int[] Right = {8,9,10,11,3,1,12,14,22,20,7,5};
	private int[] Back = {12,13,14,15,1,0,16,18,23,22,11,9};
	private int[] Left = {16,17,18,19,0,2,4,6,21,23,15,13};
	private int[] Bottom = {20,21,22,23,6,7,10,11,14,15,18,19};

	public PuzzleR() {}
	public int[] getContent() { return content;}

	public void setState(String state) {
		for (int i = 0; i < 24 ; i++){
			switch (state.charAt(i)) {
				case 'w': content[i] = 0; break;
				case 'g': content[i] = 1; break;
				case 'o': content[i] = 2; break;
				case 'b': content[i] = 3; break;
				case 'r': content[i] = 4; break;
				case 'y': content[i] = 5; break;
			}
		}
	}
	
	public boolean isInitial() {
		for (int i = 0; i < 24; i++)	if (content[i] != i / 4) return false;
		return true;
	}

	public void randomizeState(int n) {
		Random r = new Random("seed".hashCode());
		for (int i = 1; i <= n; i++) switch(r.nextInt(9)) {
			case 0: move90(Up); break;
			case 1: move180(Up); break;
			case 2: moveR90(Up); break;
			case 3: move90(Front); break;
			case 4: move180(Front); break;
			case 5: moveR90(Front); break;
			case 6: move90(Right); break;
			case 7: move180(Right); break;
			case 8: moveR90(Right); break;
			}
	}

	public void printState() {
		System.out.println("Top\tFront\tRight\tBack\tLeft\tBottom");
		for (int i = 0; i < 24; i++) {
			System.out.print(colors[content[i]] + " ");
			if (i % 4 == 1) {System.out.print("\t"); i+=2;}
		}
		System.out.println();
		for (int i = 2; i < 24; i++) {
			System.out.print(colors[content[i]] + " ");
			if (i % 4 == 3) {System.out.print("\t"); i+=2;}
		}
		System.out.println();
	}

	public boolean move(String direction) {

		if (direction.equals("U")) move90(Up);
		else if (direction.equals("U2")) move180(Up);
		else if (direction.equals("U\'")) moveR90(Up);
		else if (direction.equals("F")) move90(Front);
		else if (direction.equals("F2")) move180(Front);
		else if (direction.equals("F\'")) moveR90(Front);
		else if (direction.equals("R")) move90(Right);
		else if (direction.equals("R2")) move180(Right);
		else if (direction.equals("R\'")) moveR90(Right);
		else if (direction.equals("B")) move90(Back);
		else if (direction.equals("B2")) move180(Back);
		else if (direction.equals("B\'")) moveR90(Back);
		else if (direction.equals("L")) move90(Left);
		else if (direction.equals("L2")) move180(Left);
		else if (direction.equals("L\'")) moveR90(Left);
		else if (direction.equals("Bot")) move90(Bottom);
		else if (direction.equals("Bot2")) move180(Bottom);
		else if (direction.equals("Bot\'")) moveR90(Bottom);
		return true;
	}
	
	private void move90(int[] p) {
		circulate(p[0],p[1],p[3],p[2]);
		circulate(p[4],p[6],p[8],p[10]);
		circulate(p[5],p[7],p[9],p[11]);
	}
	private void move180(int[] p) {
		swap(p[0],p[3]);swap(p[1],p[2]);
		swap(p[4],p[8]);swap(p[6],p[10]);
		swap(p[5],p[9]);swap(p[7],p[11]);
	}
	private void moveR90(int[] p) {
		circulate(p[2],p[3],p[1],p[0]);
		circulate(p[10],p[8],p[6],p[4]);
		circulate(p[11],p[9],p[7],p[5]);
	}
	private void swap(int x, int y) {
		int t = content[x]; content[x] = content[y]; content[y] = t;
	}
	private void circulate(int x, int y, int z, int w) {
		int t = content[x];
		content[x] = content[w]; content[w] = content[z];
		content[z] = content[y]; content[y] = t;
	}

	public Puzzle clone() {
		PuzzleR p = new PuzzleR();
		for (int i = 0; i < 24 ; i++) p.content[i] = this.content[i];
		return p;
	}

	public State generateState(String moves) {
		return new StateR(this, moves);
	}
}