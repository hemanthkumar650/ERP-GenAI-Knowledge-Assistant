FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY dotnet_worker/ ./dotnet_worker/
RUN dotnet publish dotnet_worker/Worker.csproj -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/runtime:10.0
WORKDIR /app

COPY --from=build /app/publish ./

ENTRYPOINT ["dotnet", "Worker.dll"]
