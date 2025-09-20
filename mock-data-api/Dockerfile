# Use an official Node.js runtime as a parent image
FROM node:18-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# Make your port available to the world outside this container
EXPOSE 8080

# Run the app when the container launches
CMD [ "node", "index.js" ]