package com.xboxcraft.vote;

import org.bukkit.ChatColor;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;

public class VoteCommand implements CommandExecutor {
	
	private static final String voteUrl = "http://minestatus.net/26507-xboxcraft/vote";

	@Override
	public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
		sender.sendMessage(new String[] { ChatColor.GOLD + "Vote for us to get free gold! Click the link!", voteUrl });
		return true;
	}

}
