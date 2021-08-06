package com.christiandevs.entities;

import java.util.*;
import java.io.*;
import org.json.*;
import com.flume2d.graphics.*;
import com.flume2d.masks.AABB;
import com.flume2d.input.*;
import com.christiandevs.rpg.*;

public class Player extends Actor
{

	public String name;

	public Player(String name)
	{
		super(0, 0);
		
		this.layer = 5;
		this.type = "player";
		this.name = name;

		// load up the character file
		try {
			FileReader in = new FileReader("data/player/" + name + ".json");
			loadJSON(new JSONObject(new JSONTokener(in)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public void spawn(int x, int y)
	{
		this.x = x;
		this.y = y;
	}

	@Override
	public void loadJSON(JSONObject obj) throws JSONException
	{
		raceName = obj.getString("race");
		Race race = Race.load(raceName);

		sprite = race.getSpritemap();
		sprite.play("down");
		setGraphic(sprite);

		setMask(new AABB(sprite.frameWidth, sprite.frameHeight));

		super.loadJSON(obj);
	}
	
	@Override
	public void update()
	{
		switch (state)
		{
			case StartTurn:
				if (Input.check(Key.SPACE))
				{
					state = PlayState.Attack;
				}
				if (Input.touched)
				{
					Touch touch = Input.touches.get(0);
					touch.x += scene.camera.x;
					touch.y += scene.camera.y;
					//if (canMoveTo(touch.x, touch.y))
					{
						getPathTo(touch.x, touch.y, maxMoveSpaces);
						state = PlayState.Move;
					}
				}
				break;
			case Move:
				if (!followPath())
				{
					// we took our turn
					state = PlayState.Attack;
				}
				break;
			case Attack:
				if (Input.check(Key.SPACE))
				{
					state = PlayState.Wait;
				}
				if (Input.touched)
				{
					Touch touch = Input.touches.get(0);
					touch.x += scene.camera.x;
					touch.y += scene.camera.y;
					Actor enemy = (Actor) scene.findAt("monster", touch.x, touch.y);
					if (enemy != null)
					{
						attack(enemy);
						state = PlayState.Wait;
					}
				}
				break;
		}
		super.update();
	}

	private String raceName;

	private static int maxMoveSpaces = 5;
	
}
