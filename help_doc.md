`docker compose build` reads the Dockerfile and builds the image; a frozen snapshot of the OS


Two ways to use it, depending on what you're doing:

Option A — quick disposable shell (closest to what you were typing before):


docker compose run --rm reactree

Drops you into a shell, and cleans itself up when you exit — same behavior as the long command.

Option B — keep a container running in the background, hop in and out of it:


`docker compose up -d --force-recreate` restarts the cotainers from the new image.
`docker compose exec reactree bash`

The first line starts it in the background once; the second gives you a shell into it. Handy if you want to leave it running and jump back in later (e.g. across multiple terminal tabs) without restarting each time. To stop it when you're done: docker compose down.

For now, Option A is probably closest to what you want since we're still doing one-off sanity checks. Go ahead and try:


docker compose run --rm reactree


#check running comtainers
`docker ps`
`docker compose ps`