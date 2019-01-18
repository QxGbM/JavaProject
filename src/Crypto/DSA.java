package Crypto;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class DSA {
	
	private long p;
	private long m;
	
	private long g;
	private long q;
	private long x;
	private long y;
	
	private long exponent (long g, long k, long p) {
		if (k == 0) return 1;
		else return multiply(g, exponent(g, k-1, p), p);
	}
	
	private long multiply (long a, long b, long p) {
		return Math.floorMod(a * b, p);
	}
	
	private long add (long a, long b, long p) {
		return Math.floorMod(a + b, p);
	}
	
	private long inverse (long a, long m) {
		return exponent(a, euler(m)-1, m);
	}
	
	private long euler (long x) {
		ArrayList<Long> primes = sieve(x);
		ArrayList<Long> coefficients = new ArrayList<Long>();
		for (int i = 0; i < primes.size(); i++) {
			long t = 0, p = primes.get(i);
			while (Math.floorMod(x, p) == 0) {
				t++;
				x = Math.floorDiv(x, p);
			}
			coefficients.add(t);
		}
		long e = 1;
		for (int i = 0; i < primes.size(); i++) {
			long t = coefficients.get(i), p = primes.get(i);
			while(t > 1) {
				e = e * p;
				t--;
			}
			if (t == 1) {
				e = e * (p-1);
			}
		}
		return e;
	}
	
	private ArrayList<Long> sieve (long n) {
        if(n < 2) 
            return new ArrayList<Long>();
 
        ArrayList<Long> primes = new ArrayList<Long>();
        LinkedList<Long> nums = new LinkedList<Long>();
 
        for(long i = 2;i <= n;i++) {
            nums.add(i);
        }
 
        while(nums.size() > 0) {
            long nextPrime = nums.remove();
            for(long i = nextPrime * nextPrime;i <= n;i += nextPrime) {
                nums.removeFirstOccurrence(i);
            }
            primes.add(nextPrime);
        }
        return primes;
    }
	
	private long order(long a, long p) {
		for(int i = 1; i <= p-1; i++) {
			if (exponent(a, i, p) == 1) return i;
		}
		return p-1;
	}
	
	private long gcd(long a, long b) {		
		if(a == 0 || b == 0) return a + b;
		return gcd(b, Math.floorMod(a, b));
	}
	
	public long[] signDSA () {
		long[] signature = new long[2];
		Random rand = new Random();
		long k = Math.floorMod(rand.nextLong(), q);
		long R = exponent(g, k, p);
		long r = Math.floorMod(R, q);
		long s = add(multiply(k, m, q), multiply(r, x, q), q);
		signature[0] = r;
		signature[1] = s;
		System.out.println("k = " + k);
		System.out.println("Signature pair: (" + r + ", " + s + ")");
		return signature;
	}
	
	public long[] blindsignDSA () {
		
		class alice {
			@SuppressWarnings("unused")
			long kk, RR, mm;
			public long findRR() {
				Random rand = new Random();
				while(true) {
					kk = Math.floorMod(rand.nextLong(), q);
					RR = exponent(g, kk, p);
					if (gcd(RR, q) == 1) break;
				}
				System.out.println("kk = " + kk);
				System.out.println("RR = " + RR);
				return RR;
			}
			
			public long findss(long mm) {
				this.mm = mm;
				long ss = add(multiply(kk, mm, q), multiply(RR, x, q), q);
				System.out.println("ss = " + ss);
				return ss;
			}
		}
		
		class bob {
			@SuppressWarnings("unused")
			long alpha, beta, R, mm, RR, ss;
			public long findmm(long RR) {
				this.RR = RR;
				Random rand = new Random();
				if (gcd(RR, q) != 1) {
					System.out.println("error");
					System.exit(0);
				}
				while(true) {
					alpha = Math.floorMod(rand.nextLong(), q);
					beta = Math.floorMod(rand.nextLong(), q);
					R = multiply(exponent(RR, alpha, p), exponent(g, beta, p), p);
					if (gcd(R, q) == 1) break;
				}
				mm = multiply(alpha, multiply(m, multiply(RR, inverse(R, q) , q), q), q);
				System.out.println("alpha = " + alpha);
				System.out.println("beta = " + beta);
				System.out.println("R = " + R);
				System.out.println("mm = " + mm);
				return mm;
			}
			
			public long[] findrs(long ss) {
				this.ss = ss;
				long[] signature = new long[2];
				long r = Math.floorMod(R, q);
				long s = add(multiply(ss, multiply(R, inverse(RR, q), q), q), multiply(beta, m, q), q);
				signature[0] = r;
				signature[1] = s;
				System.out.println("Signature pair: (" + r + ", " + s + ")");
				return signature;
			}
		}
	
		alice a = new alice();
		bob b = new bob();
		long RR = a.findRR();
		long mm = b.findmm(RR);
		long ss = a.findss(mm);
		long[] signature = b.findrs(ss);
		return signature;
	}
	
	public void testBlindDSA() {
		
		class alice {
			@SuppressWarnings("unused")
			long x, m, kk, RR, mm;
			public long sendm(long x, long m) {
				this.x = x;
				return this.m = m;
			}
			
			public long findRR() {
				Random rand = new Random();
				while(true) {
					kk = Math.floorMod(rand.nextLong(), q);
					RR = exponent(g, kk, p);
					if (gcd(RR, q) == 1) break;
				}
				System.out.println("kk = " + kk);
				System.out.println("RR = " + RR);
				return RR;
			}
			
			public long findss(long mm) {
				this.mm = mm;
				long ss = add(multiply(kk, mm, q), multiply(RR, x, q), q);
				System.out.println("ss = " + ss);
				return ss;
			}
		}
		
		class bob {
			@SuppressWarnings("unused")
			long m, alpha, beta, R, mm, RR, ss;
			public void getm(long m) {
				this.m = m;
				System.out.println("message received: " + m);
			}
			public long findmm(long RR) {
				this.RR = RR;
				Random rand = new Random();
				if (gcd(RR, q) != 1) {
					System.out.println("error");
					System.exit(0);
				}
				while(true) {
					alpha = Math.floorMod(rand.nextLong(), q);
					beta = Math.floorMod(rand.nextLong(), q);
					R = multiply(exponent(RR, alpha, p), exponent(g, beta, p), p);
					if (gcd(R, q) == 1) break;
				}
				mm = multiply(alpha, multiply(this.m, multiply(RR, inverse(R, q) , q), q), q);
				System.out.println("alpha = " + alpha);
				System.out.println("beta = " + beta);
				System.out.println("R = " + R);
				System.out.println("mm = " + mm);
				return mm;
			}
			
			public long[] findrs(long ss) {
				this.ss = ss;
				long[] signature = new long[2];
				long r = Math.floorMod(R, q);
				long s = add(multiply(ss, multiply(R, inverse(RR, q), q), q), multiply(beta, this.m, q), q);
				signature[0] = r;
				signature[1] = s;
				System.out.println("Signature pair: (" + r + ", " + s + ")");
				return signature;
			}
		}
		
		class eve {
			@SuppressWarnings("unused")
			long fakem, truem, fakekk, fakeRR, trueRR, truemm, x, fakess;
			public long interceptm(long truem) {
				this.truem = truem;
				Random rand = new Random();
				do {
					fakem = Math.floorMod(rand.nextLong(), p);
				} while(fakem == 0 || gcd(fakem, q) != 1);
				return fakem;
			}
			
			public long interceptRR(long trueRR) {
				this.trueRR = trueRR;
				System.out.println("Intercepted RR = " + trueRR);
				Random rand = new Random();
				while(true) {
					fakekk = Math.floorMod(rand.nextLong(), q);
					fakeRR = exponent(g, fakekk, p);
					if (gcd(fakeRR, q) == 1) break;
				}
				System.out.println("Faked kk = " + fakekk);
				System.out.println("Faked RR = " + fakeRR);
				return fakeRR;
			}
			
			public long interceptmm(long truemm) {
				this.truemm = truemm;
				System.out.println("Intercepted mm = " + truemm);
				System.out.println("Faked mm = 0");
				return 0;
			}
			
			public long interceptss(long truess) {
				System.out.println("Intercepted ss = " + truess);
				x = multiply(truess, inverse(trueRR, q), q);
				System.out.println("Solved Alice's private key = " + x);
				fakess = add(multiply(fakekk, truemm, q), multiply(fakeRR, x, q), q);
				System.out.println("Faked ss = " + fakess);
				return fakess;
			}
		}
		
		alice a = new alice();
		bob b = new bob();
		eve e = new eve();
		long message = m, key = x;
		message = a.sendm(key, message);
		message = e.interceptm(message);
		b.getm(message);
		
		System.out.println("\nAlice:");
		long RR = a.findRR();
		System.out.println("\nEve:");
		RR = e.interceptRR(RR);
		System.out.println("\nBob:");
		long mm = b.findmm(RR);
		System.out.println("\nEve:");
		mm = e.interceptmm(mm);
		System.out.println("\nAlice:");
		long ss = a.findss(mm);
		System.out.println("\nEve:");
		ss = e.interceptss(ss);
		System.out.println("\nBob:");
		long[] signature = b.findrs(ss);
		System.out.println("\nAlice sent the message: " + a.m);
		System.out.println("Bob got the message: " + b.m);
		System.out.println("Bob thinks the message is " + verifyDSA(b.m, signature));
	}
	
	public boolean verifyDSA (long m, long[] signature) {
		long r = signature[0], s = signature[1];
		long T = exponent(multiply(exponent(g, s, p), exponent(inverse(y, p), r, p), p), inverse(m, q), p);
		System.out.println("T = " + T);
		return r == Math.floorMod(T, q);
	}
	
	public long[] signNyberg() {
		Random rand = new Random();
		long k = Math.floorMod(rand.nextLong(), q);
		long r = multiply(m, exponent(g, k, p), p);
		long s = add(multiply(x, r, q), k, q);
		long[] signature = {r,s};
		return signature;
	}
	
	public long[] blindsignNyberg() {
		class alice {
			@SuppressWarnings("unused")
			long kk, rr, mm;
			public long findrr() {
				Random rand = new Random();
				kk = Math.floorMod(rand.nextLong(), q);
				rr = exponent(g, kk, p);
				System.out.println("kk = " + kk);
				System.out.println("rr = " + rr);
				return rr;
			}
			
			public long findss(long mm) {
				this.mm = mm;
				long ss = add(multiply(mm, x, q), kk, q);
				System.out.println("ss = " + ss);
				return ss;
			}
		}
		
		class bob {
			@SuppressWarnings("unused")
			long alpha, beta, r, mm, rr, ss;
			public long findmm(long rr) {
				this.rr = rr;
				Random rand = new Random();
				while(true) {
					alpha = Math.floorMod(rand.nextLong(), q);
					beta = Math.floorMod(rand.nextLong(), q-1) + 1;
					r = multiply(m, multiply(exponent(g, alpha, p), exponent(rr, beta, p), p), p);
					mm = multiply(r, inverse(beta, q), q);
					if (mm > 0) break;
				}
				System.out.println("alpha = " + alpha);
				System.out.println("beta = " + beta);
				System.out.println("r = " + r);
				System.out.println("mm = " + mm);
				return mm;
			}
			
			public long[] findrs(long ss) {
				this.ss = ss;
				long[] signature = new long[2];
				long s = add(multiply(ss, beta, q), alpha, q);
				signature[0] = r;
				signature[1] = s;
				System.out.println("Signature pair: (" + r + ", " + s + ")");
				return signature;
			}
		}
		
		alice a = new alice();
		bob b = new bob();
		long rr = a.findrr();
		long mm = b.findmm(rr);
		long ss = a.findss(mm);
		long[] signature = b.findrs(ss);
		return signature;
	}
	
	public boolean verifyNyberg(long message, long[] signature) {
		long r = signature[0], s = signature[1];
		long T = multiply(exponent(inverse(g, p), s, p), multiply(exponent(y, r, p), r, p), p);
		return T == m;
	}
	
	public DSA(long m, long p) {
		this.p = p;
		this.m = m;
		q = findq(p);
		g = findg(q, p);
		x = Math.floorMod(new Random().nextLong(), q);
		y = exponent(g, x, p);
		
		System.out.println("q = " + q);
		System.out.println("g = " + g);
		System.out.println("x = " + x);
		System.out.println("y = " + y + "\n");
	}
	
	private long findq (long p) {
		long x = p-1;
		ArrayList<Long> primes = sieve(x);
		for(int i = 0; i < primes.size(); i++) {
			long t = primes.get(i);
			while (Math.floorMod(x, t) == 0) {
				if (x == t) return x;
				x = Math.floorDiv(x, t);
			}
		}
		return x;
	}
	
	private long findg (long q, long p) {
		for(long i = 1; i < p; i++) {
			if(order(i, p) == q) return i;
		}
		return 1;
	}
	
	public static void testSignDSA() {
		DSA test = new DSA(6, 23);
		long[] signature = test.signDSA();
		System.out.println(test.verifyDSA(test.m, signature));
	}
	
	public static void testBlindSignDSA() {
		DSA test = new DSA(6, 23);
		long[] signature = test.blindsignDSA();
		System.out.println(test.verifyDSA(test.m, signature));
	}
	
	public static void crackBlindDSA() {
		DSA test = new DSA(6, 23);
		test.testBlindDSA();
	}
	
	public static void main(String[] args) {
		//testSignDSA();
		//testBlindSignDSA();
		crackBlindDSA();
	}

}
