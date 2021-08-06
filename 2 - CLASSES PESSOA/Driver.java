package racingManager.model;

/**
 * A Formula 1 driver.
 * 
 */
public class Driver {
	// Information
	private String forename;
	private String lastName;
	// Settings
	private int skillLevel;
	// Championchip information
	private String teamName;
	private int points;

	private Driver(String forename, String lastName, int skillLevel,
			String teamName, int points) {
		this.forename = forename;
		this.lastName = lastName;
		this.skillLevel = skillLevel;
		this.teamName = teamName;
		this.points = points;
	}

	public Driver(String forename, String lastName, int skillLevel) {
		this.forename = forename;
		this.lastName = lastName;
		// TODO: When will team name be set ? At the moment it will never be set
		this.teamName = null;
		this.skillLevel = skillLevel;
		this.points = 0;
	}

	protected String getForename() {
		return forename;
	}

	protected String getLastName() {
		return lastName;
	}

	protected String getTeamName() {
		return teamName;
	}

	protected int getSkillLevel() {
		return skillLevel;
	}

	protected int getPoints() {
		return points;
	}

	protected void addPoints(int pointsToAdd) {
		this.points += pointsToAdd;
	}

	public String toString() {
		// TODO: Deliver all drivers information
		return skillLevel + " " + lastName;
	}

	public Driver copy(Driver driver) {
		return new Driver(forename, lastName, skillLevel, teamName, points);
	}
}