import java.util.Random;

/**
 * @author Nathan Newman
 *
 */
public class Warden {

	/**
	 * Number of prisoners in the prison.
	 */
	public static final int SIZE = 100;

	/**
	 * Integer array used to keep track of whether or not a prisoner has been
	 * selected. A 1 indicates the prisoner has and anyother number (mainly a
	 * zero indicates he/she hasn't).
	 */
	private static int[] cells = new int[SIZE];

	/**
	 * Array of prisoners used to keep track of each prisoner and what actions
	 * he has taken.
	 */
	private static Prisoner[] prison = new AsynchCon[SIZE]; // **Put any
															// implementation of
															// prisoner here.

	/**
	 * The lightBulb object used in the puzzle.
	 */
	private static LightBulb light = new LightBulb(false);

	/**
	 * Number of trials to run.
	 */
	private static final int TRIALS = 10;

	/**
	 * A method used to simulate possible strategies to the
	 * "100 Prisoners and a Light Bulb Problem". In order to test any one
	 * strategy (an implementation of prisoner) you just have to go to the four
	 * instances within the program (marked by a single line comment) and
	 * replace the current strategy with you own. The method runs the strategy a
	 * set number of trials and then at the end gives you the average number of
	 * days it took to complete (or fail) the strategy for all trials.
	 * 
	 * @param args
	 * 
	 */
	public static void main(String[] args) {
		Random gen = new Random();

		int[] numbDays = new int[Warden.TRIALS];

		// First, set the day intially to zero, and print the strategy
		// description of the given implementation of prisoner.
		int day = 0;
		Prisoner prisoner = new AsynchCon(Warden.SIZE); // **Put any
														// implementation of
														// prisoner here.
		System.out.println(prisoner.strategyDescription());

		for (int i = 0; i < Warden.TRIALS; i++) {
			// -------------------------------
			// Then, start the simulation for the given number of trials.
			boolean allVisit = false;
			while (allVisit != true) {
				// While none of the prisoners have made the assertion that all
				// have visited the room, then continue to randomly select a
				// prisoner. If that prisoner hasn't been selected before then
				// alter the cells array accordingly, otherwise call
				// prisoner.visit in order to use the strategy to check if all
				// have visited.
				int prisNumb = gen.nextInt(100);
				if (Warden.cells[prisNumb] == 0) {
					Warden.prison[prisNumb] = new AsynchCon(prisNumb); // **Put
																		// any
																		// implementation
																		// of
																		// prisoner
																		// here.
					Warden.cells[prisNumb] = 1;
				}

				allVisit = Warden.prison[prisNumb].visit(day, light);
				day++;
			}

			// The while loop is broken when one prisoner declares all have
			// visited the common room, therefore check if that assertion is
			// true.
			int count = 0;
			for (int j = 0; j < Warden.SIZE; j++) {
				if (Warden.cells[j] == 1) {
					count++;
				}
			}

			// Print a statement according to whether or not the above assertion
			// is true
			if (count != Warden.SIZE) {
				System.out.println("Bad strategy: everyone is dead!");
				// System.exit(0);
			} else {
				System.out.println("Good Job!  It took you " + day
						+ " days to solve.");
			}

			// Record the number of days it took, for statistical reason, and
			// reset all arrays, and day = 0.
			numbDays[i] = day;
			day = 0;
			Warden.cells = new int[Warden.SIZE];
			Warden.prison = new AsynchCon[Warden.SIZE]; // **Put any
														// implementation of
														// prisoner here.
			Warden.light = new LightBulb(false);
			// ------------------------------
		}

		// Find the average number of days it took for the given strategy.
		float avg = 0;
		for (int i = 0; i < Warden.TRIALS; i++) {
			avg = avg + numbDays[i];
		}
		avg = avg / 10;
		System.out
				.println("The average number of days it took for this strategy was: "
						+ avg + " days.");
	}
}
