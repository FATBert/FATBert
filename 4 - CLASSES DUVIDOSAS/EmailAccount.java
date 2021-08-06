/**
 * A class the represents email accounts provided by Ohio State. Using two
 * strings to represent the first and last name and two static arrays to
 * record/keep track of previous names entered, it is easy to generate email
 * addresses in the form lastName.#@osu.edu.
 * 
 * @mathdef IsValidString implies string contains no dashes or apostrophes
 * 
 * @correspondence fullName = firstName + lastName EmailAddress =
 *                 lastName.dotNumber@osu.edu where dotNumber is unique per the
 *                 instance of the lastName
 * 
 * 
 * @author Nathan Newman
 * 
 */
public class EmailAccount {

	/**
	 * firstName of the given student.
	 */
	private String firstName;
	
	/**
	 * lastName of the given student.
	 */
	private String lastName;
	
	/**
	 * dotNumber = unique number given to each student.
	 */
	private int dotNumber;
	
	/**
	 * list of names enetered into the student "database".
	 */
	private static String[] nameList;
	
	/**
	 * size of the student "database".
	 */
	private static int listSize;

	/**
	 * Private method that returns a unique number according to the given last name.
	 * 
	 * @param last-lastName of student
	 * 
	 * @requires IsValidString(last)
	 * 
	 * @ensures number returned by getNumber is unique
	 * 
	 * @return integer unique to given student
	 */
	private int getNumber(String last) {
		int count = 0;
		// Cycles through nameList and increases the count according to how many
		// 		occurrences of the name appear
		for (int i = 0; i < listSize; i++) {
			if (last == nameList[i]) {
				count++;
			}
		}
		return count;
	}

	/**
	 * Private method used to add the given name to the list of names.
	 * 
	 * @param last-lastName of student
	 * 
	 * @requires IsValidString(last)
	 * 
	 * @alters EmailAccount.nameList
	 * 
	 * @ensure EmailAccount = #EmailAccount.nameList + last
	 */
	private void addToList(String last) {
		// First create a tempList in order to transfer the current list over
		String[] tempList = new String[EmailAccount.listSize];
		for (int i = 0; i < EmailAccount.listSize; i++) {
			tempList[i] = EmailAccount.nameList[i];
		}
		// Increase the listSize by one
		EmailAccount.listSize++;
		// Then resize nameList, and swap tempList with nameList again
		EmailAccount.nameList = new String[EmailAccount.listSize];
		for (int i = 0; i < EmailAccount.listSize - 1; i++) {
			EmailAccount.nameList[i] = tempList[i];
		}
		// Finally add the passed in last name to nameList
		EmailAccount.nameList[EmailAccount.listSize - 1] = last;
	}

	/**
	 * Constructor that initializes the first and last names, and adds the last name to the list of names.
	 * @param first - firstName of student
	 * @param last - lastName of student
	 * 
	 * @requires IsValidString(first)
	 * 			 IsValidString(last)
	 * 
	 * @ensures firstName = first
	 * 			lastName = last
	 */
	public EmailAccount(String first, String last) {
		firstName = first;
		lastName = last;
		addToList(last);
		dotNumber = getNumber(last);
	}

	/**
	 * Returns the student's full name formatted as "firstName lastName"
	 * 
	 * @ensures getName() = firstName + " " + lastName
	 * 
	 * @return getName()
	 */
	public String getName() {
		String tempString = firstName + " " + lastName;
		return tempString;
	}

	/**
	 * Returns the student's email address formatted as "lastName.#@osu.edu"
	 * 
	 * @ensures getEmailAddress() = lastName + "." + dotNumber + "@osu.edu"
	 * 
	 * @return getEmailAddress()
	 */
	public String getEmailAddress() {
		String last = lastName.toLowerCase();
		return (last + "." + dotNumber + "@osu.edu");
	}
}
