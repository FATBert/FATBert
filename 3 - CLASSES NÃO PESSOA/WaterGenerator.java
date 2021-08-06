package me.daddychurchill.Conurbation.Plats;

import java.util.Random;

import org.bukkit.Material;

import me.daddychurchill.Conurbation.Generator;
import me.daddychurchill.Conurbation.Support.ByteChunk;

public abstract class WaterGenerator extends PlatGenerator {

	protected final static byte byteSand = (byte) Material.SAND.getId();
	protected final static byte byteDirt = (byte) Material.DIRT.getId();
	protected final static byte byteSeawall = (byte) Material.SMOOTH_BRICK.getId();
	
	public final static int shoreHeight = 3;
	public final static int seawallHeight = 2;
	protected int streetLevel;
	protected int waterLevel;
	protected int seabedLevel;
	
	private double oddsOfStairs = 0.25;
	
	public WaterGenerator(Generator noise) {
		super(noise);

		streetLevel = noise.getStreetLevel();
		waterLevel = streetLevel - shoreHeight;
		seabedLevel = noise.getSeabedLevel();
	}

	@Override
	public boolean isCompatibleEdgeChunk(PlatGenerator generator) {
		return true;
	}
	
	@Override
	public Material getGroundSurfaceMaterial(int chunkX, int chunkZ) {
		return Material.SAND;
	}
	
	protected int generateChunkColumn(ByteChunk chunk, int chunkX, int chunkZ, int blockX, int blockZ, int waterbedY) {
		chunk.setBlocks(blockX, 1, waterbedY, blockZ, byteSand);
		if (waterbedY <= waterLevel)
			chunk.setBlocks(blockX, waterbedY, waterLevel + 1, blockZ, byteWater);
		return Math.max(waterbedY, waterLevel);
	}
	
	protected void generateSeawalls(ByteChunk chunk, Random random, int chunkX, int chunkZ) {
		boolean toNorth = !noise.isWater(chunkX, chunkZ - 1);
		boolean toSouth = !noise.isWater(chunkX, chunkZ + 1);
		boolean toWest = !noise.isWater(chunkX - 1, chunkZ);
		boolean toEast = !noise.isWater(chunkX + 1, chunkZ);
		
		if (toNorth && toWest && !toSouth && !toEast) {
			PlatGenerator neighborPlat = noise.getNeighboringPlatGenerator(chunkX, chunkZ, -1, -1);
			for (int x = 0; x < 16; x++) {
				for (int z = 0; z < 16 - x; z++) {
					int y = neighborPlat.generateChunkColumn(chunk, chunkX, chunkZ, x, z);
					if (z == 16 - x - 1)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight, z, byteSeawall);
				}
			}
		} else if (toNorth && toEast && !toSouth && !toWest) {
			PlatGenerator neighborPlat = noise.getNeighboringPlatGenerator(chunkX, chunkZ, +1, -1);
			for (int z = 0; z < 16; z++) {
				for (int x = z; x < 16; x++) {
					int y = neighborPlat.generateChunkColumn(chunk, chunkX, chunkZ, x, z);
					if (x == z)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight, z, byteSeawall);
				}
			}
		} else if (toSouth && toWest && !toNorth && !toEast) {
			PlatGenerator neighborPlat = noise.getNeighboringPlatGenerator(chunkX, chunkZ, -1, +1);
			for (int x = 0; x < 16; x++) {
				for (int z = x; z < 16; z++) {
					int y = neighborPlat.generateChunkColumn(chunk, chunkX, chunkZ, x, z);
					if (z == x)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight, z, byteSeawall);
				}
			}
		} else if (toSouth && toEast && !toNorth && !toWest) {
			PlatGenerator neighborPlat = noise.getNeighboringPlatGenerator(chunkX, chunkZ, +1, +1);
			for (int z = 0; z < 16; z++) {
				for (int x = 15 - z; x < 16; x++) {
					int y = neighborPlat.generateChunkColumn(chunk, chunkX, chunkZ, x, z);
					if (x == 15 - z)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight, z, byteSeawall);
				}
			}
		} else {
			boolean isBridge = noise.isRoad(chunkX, chunkZ);
			if (toNorth) {
				boolean addStairs = isStairs(random);
				PlatGenerator neighborPlat = noise.getTopPlatGenerator(chunkX, chunkZ - 1);
				int seawallOffset = isBridge && noise.isRoad(chunkX, chunkZ - 1) ? -RoadGenerator.roadThickness : 0;
				for (int x = 0; x < 16; x++) {
					int y = neighborPlat.getGroundSurfaceY(chunkX, chunkZ - 1, x, 15);
					if (addStairs && x == 1)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset - 1, 0, byteSeawall);
					else
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset, 0, byteSeawall);
					if (addStairs && x > 0 && x < 14)
						chunk.setBlock(x, y + seawallHeight + seawallOffset - x - 1, 1, byteSeawall);
					if (x == 0 || x == 15)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset, 1, byteSeawall);
				}
			} 
			if (toSouth) {
				boolean addStairs = isStairs(random);
				PlatGenerator neighborPlat = noise.getTopPlatGenerator(chunkX, chunkZ + 1);
				int seawallOffset = isBridge && noise.isRoad(chunkX, chunkZ + 1) ? -RoadGenerator.roadThickness : 0;
				for (int x = 0; x < 16; x++) {
					int y = neighborPlat.getGroundSurfaceY(chunkX, chunkZ + 1, x, 0);
					if (addStairs && x == 1)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset - 1, 15, byteSeawall);
					else
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset, 15, byteSeawall);
					if (addStairs && x > 0 && x < 14)
						chunk.setBlock(x, y + seawallHeight + seawallOffset - x - 1, 14, byteSeawall);
					if (x == 0 || x == 15)
						chunk.setBlocks(x, seabedLevel, y + seawallHeight + seawallOffset, 14, byteSeawall);
				}
			} 
			if (toWest) {
				boolean addStairs = isStairs(random);
				PlatGenerator neighborPlat = noise.getTopPlatGenerator(chunkX - 1, chunkZ);
				int seawallOffset = isBridge && noise.isRoad(chunkX - 1, chunkZ) ? -RoadGenerator.roadThickness : 0;
				for (int z = 0; z < 16; z++) {
					int y = neighborPlat.getGroundSurfaceY(chunkX - 1, chunkZ, 15, z);
					if (addStairs && z == 1)
						chunk.setBlocks(0, seabedLevel, y + seawallHeight + seawallOffset - 1, z, byteSeawall);
					else
						chunk.setBlocks(0, seabedLevel, y + seawallHeight + seawallOffset, z, byteSeawall);
					if (addStairs && z > 0 && z < 14)
						chunk.setBlock(1, y + seawallHeight + seawallOffset - z - 1, z, byteSeawall);
					if (z == 0 || z == 15)
						chunk.setBlocks(1, seabedLevel, y + seawallHeight + seawallOffset, z, byteSeawall);
				}
			} 
			if (toEast) {
				boolean addStairs = isStairs(random);
				PlatGenerator neighborPlat = noise.getTopPlatGenerator(chunkX + 1, chunkZ);
				int seawallOffset = isBridge && noise.isRoad(chunkX - 1, chunkZ) ? -RoadGenerator.roadThickness : 0;
				for (int z = 0; z < 16; z++) {
					int y = neighborPlat.getGroundSurfaceY(chunkX + 1, chunkZ, 0, z);
					if (addStairs && z == 1)
						chunk.setBlocks(15, seabedLevel, y + seawallHeight + seawallOffset - 1, z, byteSeawall);
					else
						chunk.setBlocks(15, seabedLevel, y + seawallHeight + seawallOffset, z, byteSeawall);
					if (addStairs && z > 0 && z < 14)
						chunk.setBlock(14, y + seawallHeight + seawallOffset - z - 1, z, byteSeawall);
					if (z == 0 || z == 15)
						chunk.setBlocks(14, seabedLevel, y + seawallHeight + seawallOffset, z, byteSeawall);
				}
			}
		}
	}
	
	private boolean isStairs(Random random) {
		return random.nextDouble() < oddsOfStairs;
	}
}
