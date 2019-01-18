package AI;
import java.util.LinkedList;
import java.util.Scanner;
 
public class Sieve_Method
{
    public static LinkedList<Integer> sieve(int n)
    {
        if(n < 2) 
            return new LinkedList<Integer>();
 
        LinkedList<Integer> primes = new LinkedList<Integer>();
        LinkedList<Integer> nums = new LinkedList<Integer>();
 
        for(int i = 2;i <= n;i++)
        { //unoptimized
            nums.add(i);
        }
 
        while(nums.size() > 0)
        {
            int nextPrime = nums.remove();
            for(int i = nextPrime * nextPrime;i <= n;i += nextPrime)
            {
                nums.removeFirstOccurrence(i);
            }
            primes.add(nextPrime);
        }
        return primes;
    }
    public static void main(String args[])
    {
        System.out.println("Enter the upper bound : ");
        Scanner sc = new Scanner(System.in);
        int end = sc.nextInt();
 
        System.out.println(sieve(end));
        sc.close();
        
        
 
    }
}